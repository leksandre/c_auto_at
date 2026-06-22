#!/bin/bash
# ============================================================================
# stable_rl_launcher.sh — Stable RL Training with Auto-Recovery
# ============================================================================
# Runs INSIDE the container. Manages full pipeline with auto-restart.
# Key features:
#   - Auto-detects training death → restarts pipeline + resumes from checkpoint
#   - Auto-detects EKF degradation (altitude > 50m) → full pipeline restart
#   - Auto-detects drone stuck (no new episodes) → full restart
#   - Trains from BC weights, resumes from checkpoints
# Usage:
#   bash stable_rl_launcher.sh              # full launch
#   bash stable_rl_launcher.sh --resume PATH # resume from checkpoint
#   bash stable_rl_launcher.sh --quick       # quick test (100k steps)
# ============================================================================

set +u
set +e

# ── Config ────────────────────────────────────────────────────────────────
MAX_PIPELINE_RESTARTS=500
PX4_LOG="/tmp/px4.log"
MICROXRCE_LOG="/tmp/microxrce.log"
LIGHTTRACK_LOG="/tmp/lighttrack.log"
RL_PIPELINE_LOG="/tmp/rl_pipeline.log"
TRAIN_LOG="/tmp/rl_train.log"
LOG_DIR="/tmp/rl_stable"
DRONE_POSE=""

# ── Parse args ────────────────────────────────────────────────────────────
RESUME_PATH=""
QUICK_MODE=false
while [ $# -gt 0 ]; do
    case "$1" in
        --resume) RESUME_PATH="$2"; shift 2 ;;
        --quick) QUICK_MODE=true; shift ;;
        *) shift ;;
    esac
done

mkdir -p "$LOG_DIR"

# ── Source ROS ────────────────────────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
source /simulation_ws/install/setup.bash

# ── Functions ─────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/launcher.log"
}

kill_all() {
    log "Killing all processes..."
    pkill -9 -f 'train_rl' 2>/dev/null; true
    pkill -9 -f 'px4 ' 2>/dev/null; true
    pkill -9 -f 'gz sim' 2>/dev/null; true
    pkill -9 -f 'ros2 launch' 2>/dev/null; true
    pkill -9 -f 'parameter_bridge' 2>/dev/null; true
    pkill -9 -f 'MicroXRCE' 2>/dev/null; true
    pkill -9 -f 'lighttrack' 2>/dev/null; true
    pkill -9 -f 'LightTrackNode' 2>/dev/null; true
    pkill -9 -f 'gazebo_reset' 2>/dev/null; true
    pkill -9 -f 'moving_target' 2>/dev/null; true
    sleep 3
    rm -f /dev/shm/* /tmp/cyclone_* /tmp/fastrtps_* /tmp/rmw_* 2>/dev/null
    log "All processes killed + shared memory cleaned"
}

# ── Trap for graceful shutdown ──────────────────────────────────────────
cleanup() {
    log "Launcher shutting down (signal=$1)..."
    kill_all
    exit 0
}
trap 'cleanup SIGTERM' SIGTERM
trap 'cleanup SIGINT' SIGINT
trap 'cleanup EXIT' EXIT

wait_for_px4() {
    log "Waiting for PX4 to be ready..."
    for i in $(seq 1 120); do
        if strings "$PX4_LOG" 2>/dev/null | grep -q 'Ready for takeoff'; then
            log "PX4 ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    log "WARNING: PX4 not ready after 120s"
    return 1
}

check_px4_alive() {
    pgrep -f 'px4 ' >/dev/null 2>&1 && pgrep -f 'gz sim' >/dev/null 2>&1
}

check_training_alive() {
    pgrep -f 'train_rl' >/dev/null 2>&1
}

check_dds_alive() {
    pgrep -f 'MicroXRCE' >/dev/null 2>&1
}

get_episode_count() {
    local count
    count=$(strings "$TRAIN_LOG" 2>/dev/null | grep -c 'Episode reset complete' || echo "0")
    # grep -c can return two numbers if strings splits the file
    count=$(echo "$count" | tail -1)
    echo "${count:-0}"
}

get_ekf_altitude() {
    # Get last known altitude from training log
    local alt
    alt=$(strings "$TRAIN_LOG" 2>/dev/null | grep -oP 'Alt:\s+\K[0-9.]+' | tail -1)
    echo "${alt:-0}"
}

get_arm_failures() {
    local count
    count=$(strings "$TRAIN_LOG" 2>/dev/null | grep -c 'Failed to arm after' || echo "0")
    count=$(echo "$count" | tail -1)
    echo "${count:-0}"
}

start_px4() {
    log "Starting PX4 SITL + Gazebo (NVIDIA GPU)..."
    # Random drone pose: 30-40m from car at [50, 0, 0], yaw toward car
    local angle=$(python3 -c "import random,math; a=random.uniform(0,2*math.pi); d=random.uniform(30,40); cx=50.0; x=cx+d*math.cos(a); y=d*math.sin(a); yaw=math.atan2(-y,-x); print(f'{x:.2f},{y:.2f},0.0,0.0,0.0,{yaw:.4f}')")
    DRONE_POSE="$angle"
    log "Drone pose: $DRONE_POSE"
    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia nohup ros2 launch px4_bringup px4_sim_attack_car.launch.py "model_pose:=$DRONE_POSE" > "$PX4_LOG" 2>&1 &
}

start_dds() {
    log "Starting DDS bridge..."
    nohup MicroXRCEAgent udp4 -p 8888 > "$MICROXRCE_LOG" 2>&1 &
}

start_lighttrack() {
    log "Starting LightTrack..."
    export QT_QPA_PLATFORM=offscreen
    nohup ros2 launch lighttrack_ros lighttrack_launch.py > "$LIGHTTRACK_LOG" 2>&1 &
}

start_rl_pipeline() {
    log "Starting RL Pipeline..."
    local drone_pose_arg=""
    if [ -n "$DRONE_POSE" ]; then
        drone_pose_arg="drone_pose:=$DRONE_POSE"
    fi
    nohup ros2 launch los_guidance rl_pipeline_launch.py $drone_pose_arg > "$RL_PIPELINE_LOG" 2>&1 &
}

start_training() {
    local resume_arg=""
    if [ -n "$RESUME_PATH" ] && [ -f "$RESUME_PATH" ]; then
        resume_arg="--resume $RESUME_PATH"
        log "Resuming from: $RESUME_PATH"
    fi

    local quick_arg=""
    if [ "$QUICK_MODE" = true ]; then
        quick_arg="--quick"
    fi

    log "Starting RL Training..."
    source /simulation_ws/.venv/bin/activate
    cd /simulation_ws/src/los_guidance/rl
    nohup python3 train_rl.py \
        --bc-weights /simulation_ws/onnx_model/best_guidance_mlp.pth \
        --bc-model /simulation_ws/onnx_model/guidance_mlp.onnx \
        $resume_arg $quick_arg \
        > "$TRAIN_LOG" 2>&1 &
    log "Training started (PID: $!)"
}

find_latest_checkpoint() {
    ls -t /simulation_ws/src/los_guidance/logs/rl_checkpoint_*.zip 2>/dev/null | head -1
}

full_pipeline_start() {
    kill_all

    log "=== Starting full pipeline ==="
    start_px4
    wait_for_px4

    start_dds
    sleep 5

    start_lighttrack
    sleep 3

    start_rl_pipeline
    sleep 3

    log "=== Pipeline started ==="
    sleep 10
}

# ── Main Loop ─────────────────────────────────────────────────────────────

RESTART_COUNT=0
LAST_RESTART_TIME=0
LAST_EPISODE_COUNT=0
LAST_EPISODE_CHANGE_TIME=$(date +%s)
LAST_ARM_FAILURES=0

log "============================================="
log "  Stable RL Training Launcher v2"
log "  Resume: ${RESUME_PATH:-none}"
log "  Quick: $QUICK_MODE"
log "============================================="

full_pipeline_start
start_training

while [ $RESTART_COUNT -lt $MAX_PIPELINE_RESTARTS ]; do
    sleep 15

    # ── Check: training alive? ────────────────────────────────────────
    if ! check_training_alive; then
        log "━━━ Training died ━━━"
        tail -5 "$TRAIN_LOG" 2>/dev/null | tee -a "$LOG_DIR/launcher.log"

        RESTART_COUNT=$((RESTART_COUNT + 1))
        log "Pipeline restart #$RESTART_COUNT"
        full_pipeline_start

        LATEST_CKPT=$(find_latest_checkpoint)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_PATH="$LATEST_CKPT"
        fi
        start_training
        LAST_RESTART_TIME=$(date +%s)
        LAST_EPISODE_COUNT=0
        LAST_ARM_FAILURES=0
        LAST_EPISODE_CHANGE_TIME=$(date +%s)
        continue
    fi

    # ── Check: PX4 alive? ────────────────────────────────────────────
    if ! check_px4_alive; then
        log "━━━ PX4 died ━━━"
        RESTART_COUNT=$((RESTART_COUNT + 1))
        log "Pipeline restart #$RESTART_COUNT"
        full_pipeline_start

        LATEST_CKPT=$(find_latest_checkpoint)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_PATH="$LATEST_CKPT"
        fi
        start_training
        LAST_RESTART_TIME=$(date +%s)
        LAST_EPISODE_COUNT=0
        LAST_ARM_FAILURES=0
        LAST_EPISODE_CHANGE_TIME=$(date +%s)
        continue
    fi

    # ── Check: DDS alive? ────────────────────────────────────────────
    if ! check_dds_alive; then
        log "DDS died — restarting..."
        start_dds
        sleep 5
    fi

    # ── Check: EKF divergence (altitude > 50m or < -2m) ──────────────
    ALT=$(get_ekf_altitude)
    if [ "$(echo "$ALT > 50" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        log "━━━ EKF DIVERGENCE: altitude=${ALT}m > 50m ━━━"
        RESTART_COUNT=$((RESTART_COUNT + 1))
        log "Pipeline restart #$RESTART_COUNT"
        full_pipeline_start

        LATEST_CKPT=$(find_latest_checkpoint)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_PATH="$LATEST_CKPT"
        fi
        start_training
        LAST_RESTART_TIME=$(date +%s)
        LAST_EPISODE_COUNT=0
        LAST_ARM_FAILURES=0
        LAST_EPISODE_CHANGE_TIME=$(date +%s)
        continue
    fi
    if [ "$(echo "$ALT < -2" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        log "━━━ EKF DIVERGENCE: altitude=${ALT}m < -2m (underground) ━━━"
        RESTART_COUNT=$((RESTART_COUNT + 1))
        log "Pipeline restart #$RESTART_COUNT"
        full_pipeline_start

        LATEST_CKPT=$(find_latest_checkpoint)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_PATH="$LATEST_CKPT"
        fi
        start_training
        LAST_RESTART_TIME=$(date +%s)
        LAST_EPISODE_COUNT=0
        LAST_ARM_FAILURES=0
        LAST_EPISODE_CHANGE_TIME=$(date +%s)
        continue
    fi

    # ── Check: arm failures increasing (EKF degraded) ───────────────
    ARM_FAILS=$(get_arm_failures)
    if [ "$ARM_FAILS" -gt "$LAST_ARM_FAILURES" ] && [ "$ARM_FAILS" -gt 2 ]; then
        log "━━━ EKF DEGRADED: $ARM_FAILS arm failures ━━━"
        RESTART_COUNT=$((RESTART_COUNT + 1))
        log "Pipeline restart #$RESTART_COUNT"
        full_pipeline_start

        LATEST_CKPT=$(find_latest_checkpoint)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_PATH="$LATEST_CKPT"
        fi
        start_training
        LAST_RESTART_TIME=$(date +%s)
        LAST_EPISODE_COUNT=0
        LAST_ARM_FAILURES=0
        LAST_EPISODE_CHANGE_TIME=$(date +%s)
        continue
    fi
    LAST_ARM_FAILURES=$ARM_FAILS

    # ── Check: stuck (no new episodes for 3 min) ────────────────────
    EPISODE_COUNT=$(get_episode_count)
    NOW=$(date +%s)
    if [ "$EPISODE_COUNT" -eq "$LAST_EPISODE_COUNT" ] && [ "$EPISODE_COUNT" -gt 0 ]; then
        STUCK_TIME=$((NOW - LAST_EPISODE_CHANGE_TIME))
        if [ $STUCK_TIME -gt 180 ]; then
            log "━━━ STUCK: no episodes for ${STUCK_TIME}s (count=$EPISODE_COUNT) ━━━"
            RESTART_COUNT=$((RESTART_COUNT + 1))
            full_pipeline_start

            LATEST_CKPT=$(find_latest_checkpoint)
            if [ -n "$LATEST_CKPT" ]; then
                RESUME_PATH="$LATEST_CKPT"
            fi
            start_training
            LAST_RESTART_TIME=$NOW
            LAST_EPISODE_COUNT=0
            LAST_ARM_FAILURES=0
            LAST_EPISODE_CHANGE_TIME=$NOW
            continue
        fi
    elif [ "$EPISODE_COUNT" -ne "$LAST_EPISODE_COUNT" ]; then
        LAST_EPISODE_COUNT=$EPISODE_COUNT
        LAST_EPISODE_CHANGE_TIME=$NOW
    fi

    # ── Heartbeat ────────────────────────────────────────────────────
    PID=$(pgrep -f 'train_rl' | head -1)
    log "♥ alive (PID=$PID, ep=$EPISODE_COUNT, alt=${ALT}m, arm_fails=$ARM_FAILS, restarts=$RESTART_COUNT)"

done

log "ERROR: Max restarts ($MAX_PIPELINE_RESTARTS) reached."
kill_all
exit 1
