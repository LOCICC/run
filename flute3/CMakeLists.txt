################################################################################
## BSD 3-Clause License
##
## Copyright (c) 2018, Iowa State University All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## * Redistributions of source code must retain the above copyright notice,
## this list of conditions and the following disclaimer.
##
## * Redistributions in binary form must reproduce the above copyright notice,
## this list of conditions and the following disclaimer in the documentation
## and/or other materials provided with the distribution.
##
## * Neither the name of the copyright holder nor the names of its contributors
## may be used to endorse or promote products derived from this software
## without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
## USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################

CMAKE_MINIMUM_REQUIRED (VERSION 3.1)
PROJECT(flute)

SET(CMAKE_CXX_STANDARD 11)

SET(FLUTE_HOME ${PROJECT_SOURCE_DIR})

set(POWV5_DAT ${FLUTE_HOME}/etc/POWV5.dat)
set(POST5_DAT ${FLUTE_HOME}/etc/POST5.dat)
set(POWV5_CPP ${CMAKE_CURRENT_BINARY_DIR}/POWV5.cpp)
set(POST5_CPP ${CMAKE_CURRENT_BINARY_DIR}/POST5.cpp)

add_custom_command(OUTPUT ${POWV5_CPP}
  COMMAND ${FLUTE_HOME}/etc/MakeDatVar.tcl powv5 ${POWV5_CPP} ${POWV5_DAT}
  DEPENDS ${POWV5_DAT} ${FLUTE_HOME}/etc/MakeDatVar.tcl
  )

add_custom_command(OUTPUT ${POST5_CPP}
  COMMAND ${FLUTE_HOME}/etc/MakeDatVar.tcl post5 ${POST5_CPP} ${POST5_DAT}
  DEPENDS ${POST5_DAT} ${FLUTE_HOME}/etc/MakeDatVar.tcl
  )

ADD_LIBRARY(flute
  STATIC
  flute.cpp
  ${POWV5_CPP}
  ${POST5_CPP}
  )

TARGET_INCLUDE_DIRECTORIES(flute PUBLIC ${FLUTE_HOME})
# CMakeLists.txt 极端优化
target_compile_options(flute PRIVATE
    -O3 -march=native -ffast-math
)