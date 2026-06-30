#!/bin/bash

kerType=$(uname -m)
if [ "${kerType}" == "x86_64" ]; then
	libsPath="../lib/GLNXA64"
elif [ "${kerType}" == "i686" ]; then
    libsPath="../lib/GLNX86"
else
	echo "Something is wrong with system identification"
fi

# release
cd executable
g++ -o3 -Wall ../J-UNIWARD_src/J-UNIWARD.cpp ../J-UNIWARD_src/cost_model_config.cpp ../J-UNIWARD_src/cost_model.cpp ../include/base_cost_model.cpp ../include/base_cost_model_config.cpp ../include/mi_embedder.cpp ../include/stc_ml_c.cpp ../include/stc_embed_c.cpp ../include/stc_extract_c.cpp ../include/common.cpp ../include/jstruct.cpp ../include/info_theory.cpp -o J-UNIWARD -I../include -L${libsPath} -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt ../lib/libjpeg.a