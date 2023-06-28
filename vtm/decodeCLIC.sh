#!/bin/sh

#./vtm/bin/EncoderCLIC -c ./vtm/cfg/encoder_lowdelay_clic.cfg -c ./vtm/cfg/per-sequence/50.cfg -q $2 -i $1_org.yuv -wdt $3 -hgt $4 -wdtc $5 -hgtc $6 -o $1_rec.yuv -b $1.bin >>$1.log 


./vtm/bin/DecoderCLIC -b $1.bin -o $1_dec.yuv > $1_dec.log

# -c ./vtm/cfg/encoder_lowdelay_clic.cfg -c ./vtm/cfg/per-sequence/50.cfg -q $2 -i $1_org.yuv -wdt $3 -hgt $4 -wdtc $5 -hgtc $6   >>$1.log 


# ./vtm/bin/VTM10Dec -b $1.bin -o $1_dec.rgb --OutputColourSpaceConvert=GBRtoRGB > $1_dec.log
