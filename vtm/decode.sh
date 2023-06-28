#!/bin/sh


./vtm/bin/VTM10Dec -b $1.bin -o $1_dec.rgb --OutputColourSpaceConvert=GBRtoRGB > $1_dec.log

