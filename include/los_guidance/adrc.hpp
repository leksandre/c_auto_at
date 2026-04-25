/* BSD 3-Clause License 
 * Copyright (c) 2023, snowpang 
 * All rights reserved. 

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.

 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.

 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

/**
* @author Snow Pang,ssnowpang@gmail.com
* @date 2025-11
*/

#ifndef LOS_GUIDANCE_ADRC_HPP_
#define LOS_GUIDANCE_ADRC_HPP_

#pragma once

#include <deque>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>
#include <limits>
#include <map>
#include <string>
#include <cmath>
#include <iostream>



typedef struct {
  double x1;  // filter
  double x2;  // diff
  double r;   // track ratio
  double h0;  // filter coeff
  double h;   // update step
} TD_TypeDef;

class ADRC {
public:
    ADRC(){}

    void init_td_params(double r, double h, double td_n){
      td_.x1 = 0;
      td_.x2 = 0;
      td_.r = r;
      td_.h = h;
      td_.h0 = td_n*h;
    }

    void fhan_td(double x){
      double x1_delta = td_.x1 - x; //x1-v(k)
      double d = td_.r * td_.h0 * td_.h0; //d = rh^2 ,用h0代替h,解决最速跟踪微分器速度超调问题
      double a0 = td_.h0 * td_.x2; //a0 = h*x2
      double y = x1_delta + a0; //y = x1 + a0
      double a1 = sqrt(d * (d + 8*abs(y))); // a1=sqrt(d*(d+8*ABS(y))])
      double a2 = a0 + sign(y) * (a1 - d)/2.0; // a2 = a0 + sign(y)*(a1-d)/2
      double a = (a0 + y) * fsg(y , d) + a2 * (1 - fsg(y, d));
      double fh = -td_.r * (a/d)*fsg(a,d) - td_.r * sign(a) * (1 - fsg(a, d)); //fh = -r*(a/d)*sgn(a) - r*sgn(a)*(1-sgn(a))
      td_.x1 += td_.h * td_.x2; //x1 = x1' + h*x2'
      td_.x2 += td_.h * fh; //x2 = x2' + h*fh
    }

    double get_td_x1(){
      return td_.x1;
    }

    double get_td_x2(){
      return td_.x2;
    }

private:
  // sign function
  inline int sign(double in){
    int out = 0;
    if (in>1E-6) {
      out = 1;
    } else if (in<-1E-6) {
      out = -1;
    }
    return out;
  }

  inline int fsg(double x, double d) {
    int out = 0;
	  out = (sign(x + d) - sign(x - d)) / 2;
	  return out;
  }

  TD_TypeDef td_;

};

#endif  // LOS_GUIDANCE_ADRC_HPP_