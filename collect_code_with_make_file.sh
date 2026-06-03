#!/bin/bash

# ============================================================================
# Скрипт собирает исходный код и информацию о сборке из:
#   - ./src
#   - ./include/los_guidance
#   - ./config (YAML файлы)
#   - ./CMakeLists.txt (и другие файлы сборки)
# Расширения: .cpp, .hpp, .txt, .yaml, .yml, CMakeLists.txt
# ============================================================================

OUTPUT_FILE="project_code.txt"

# Очищаем выходной файл
> "$OUTPUT_FILE"

# Заголовок
{
    echo "================================================================================"
    echo "СОБРАННЫЙ ИСХОДНЫЙ КОД И ИНФОРМАЦИЯ О СБОРКЕ"
    echo "Дата: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Директория: $(pwd)"
    echo "================================================================================"
    echo ""
} >> "$OUTPUT_FILE"

file_count=0
total_lines=0
makefile_count=0
yaml_count=0

# ============================================================================
# СЕКЦИЯ 1: Файлы из ./include/los_guidance
# ============================================================================

if [ -d "./include/los_guidance" ]; then
    echo "=== include/los_guidance ===" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    while IFS= read -r file; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            {
                echo "================================================================================"
                echo "Файл: $file"
                echo "Строк: $lines"
                echo "================================================================================"
                echo ""
                cat "$file"
                echo ""
            } >> "$OUTPUT_FILE"

            ((file_count++))
            ((total_lines += lines))
            echo "✓ $file ($lines строк)"
        fi
    done < <(find ./include/los_guidance -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.txt" -o -name "*.h" \) | sort)
else
    echo "⚠ Директория ./include/los_guidance не найдена"
fi

echo "" >> "$OUTPUT_FILE"

# ============================================================================
# СЕКЦИЯ 2: Файлы из ./src
# ============================================================================

if [ -d "./src" ]; then
    echo "=== src ===" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    while IFS= read -r file; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            {
                echo "================================================================================"
                echo "Файл: $file"
                echo "Строк: $lines"
                echo "================================================================================"
                echo ""
                cat "$file"
                echo ""
            } >> "$OUTPUT_FILE"

            ((file_count++))
            ((total_lines += lines))
            echo "✓ $file ($lines строк)"
        fi
    done < <(find ./src -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.txt" -o -name "*.h" \) | sort)
else
    echo "⚠ Директория ./src не найдена"
fi

echo "" >> "$OUTPUT_FILE"

# ============================================================================
# СЕКЦИЯ 3: YAML конфигурационные файлы (НОВОЕ!)
# ============================================================================

echo "=== YAML КОНФИГУРАЦИОННЫЕ ФАЙЛЫ ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Ищем файлы: *.yaml, *.yml в директории config
while IFS= read -r yaml_file; do
    if [ -f "$yaml_file" ]; then
        lines=$(wc -l < "$yaml_file")
        {
            echo "================================================================================"
            echo "Файл конфигурации: $yaml_file"
            echo "Строк: $lines"
            echo "================================================================================"
            echo ""
            cat "$yaml_file"
            echo ""
        } >> "$OUTPUT_FILE"

        ((file_count++))
        ((total_lines += lines))
        ((yaml_count++))
        echo "✓ $yaml_file ($lines строк)"
    fi
done < <(find ./config -maxdepth 2 -type f \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | sort)

# Также ищем в корневой директории
while IFS= read -r yaml_file; do
    if [ -f "$yaml_file" ]; then
        lines=$(wc -l < "$yaml_file")
        {
            echo "================================================================================"
            echo "Файл конфигурации: $yaml_file"
            echo "Строк: $lines"
            echo "================================================================================"
            echo ""
            cat "$yaml_file"
            echo ""
        } >> "$OUTPUT_FILE"

        ((file_count++))
        ((total_lines += lines))
        ((yaml_count++))
        echo "✓ $yaml_file ($lines строк)"
    fi
done < <(find . -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | sort)

if [ $yaml_count -eq 0 ]; then
    echo "⚠ YAML файлы не найдены" >> "$OUTPUT_FILE"
    echo "⚠ YAML файлы не найдены"
fi

echo "" >> "$OUTPUT_FILE"

# ============================================================================
# СЕКЦИЯ 4: Файлы сборки (CMakeLists.txt, Makefile)
# ============================================================================

echo "=== ФАЙЛЫ СБОРКИ ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Ищем файлы: CMakeLists.txt, Makefile, *.mk
while IFS= read -r makefile; do
    if [ -f "$makefile" ]; then
        lines=$(wc -l < "$makefile")
        {
            echo "================================================================================"
            echo "Файл сборки: $makefile"
            echo "Строк: $lines"
            echo "================================================================================"
            echo ""

            # --- Извлечение переменных сборки (Аналитика) ---
            echo "--- КЛЮЧЕВЫЕ ПЕРЕМЕННЫЕ СБОРКИ ---"

            # Ищем стандартные переменные
            grep -E "^[[:space:]]*(CC|CXX|CFLAGS|CXXFLAGS|LDFLAGS|TARGET|OUTPUT|project_SOURCE_FILES|project_DEPENDENCIES)[[:space:]]*[:]?=" "$makefile" | head -10

            echo ""
            echo "--- СОДЕРЖИМОЕ ФАЙЛА ---"
            echo ""
            cat "$makefile"
            echo ""
        } >> "$OUTPUT_FILE"

        ((file_count++))
        ((total_lines += lines))
        ((makefile_count++))
        echo "✓ $makefile ($lines строк)"
    fi
done < <(find . -maxdepth 3 -type f \( -name "CMakeLists.txt" -o -name "Makefile" -o -name "makefile" -o -name "*.mk" \) | sort)

if [ $makefile_count -eq 0 ]; then
    echo "⚠ Файлы сборки не найдены" >> "$OUTPUT_FILE"
    echo "⚠ Файлы сборки не найдены"
fi

echo "" >> "$OUTPUT_FILE"

# ============================================================================
# СЕКЦИЯ 5: Launch файлы (НОВОЕ!)
# ============================================================================

echo "=== LAUNCH ФАЙЛЫ ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

launch_count=0

while IFS= read -r launch_file; do
    if [ -f "$launch_file" ]; then
        lines=$(wc -l < "$launch_file")
        {
            echo "================================================================================"
            echo "Launch файл: $launch_file"
            echo "Строк: $lines"
            echo "================================================================================"
            echo ""
            cat "$launch_file"
            echo ""
        } >> "$OUTPUT_FILE"

        ((file_count++))
        ((total_lines += lines))
        ((launch_count++))
        echo "✓ $launch_file ($lines строк)"
    fi
done < <(find ./launch -maxdepth 2 -type f \( -name "*.py" -o -name "*.xml" -o -name "*.launch" \) 2>/dev/null | sort)

if [ $launch_count -eq 0 ]; then
    echo "⚠ Launch файлы не найдены" >> "$OUTPUT_FILE"
    echo "⚠ Launch файлы не найдены"
fi

echo "" >> "$OUTPUT_FILE"

# ============================================================================
# Итоги
# ============================================================================

{
    echo ""
    echo "================================================================================"
    echo "СТАТИСТИКА"
    echo "================================================================================"
    echo "Всего файлов (код + конфиги + сборка): $file_count"
    echo "Из них файлов сборки: $makefile_count"
    echo "Из них YAML конфигов: $yaml_count"
    echo "Из них Launch файлов: $launch_count"
    echo "Всего строк: $total_lines"
    echo "================================================================================"
} >> "$OUTPUT_FILE"

echo ""
echo "========================================"
echo "✅ Готово!"
echo "Файлов: $file_count"
echo "  - Код: $((file_count - makefile_count - yaml_count - launch_count))"
echo "  - YAML конфиги: $yaml_count"
echo "  - Launch файлы: $launch_count"
echo "  - Файлы сборки: $makefile_count"
echo "Строк: $total_lines"
echo "Выходной файл: $OUTPUT_FILE"
echo "========================================"