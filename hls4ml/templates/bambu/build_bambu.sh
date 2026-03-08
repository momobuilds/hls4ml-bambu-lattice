#!/bin/bash
set -e
insert_bambu_command

# copy final reports
src_root="HLS_output/Synthesis/vivado_flow"
dst_root="vivado_reports"
mkdir -p "$dst_root"
find "$src_root" -type f \( -iname "*.rpt" -o -iname "*.xml" \) -exec cp -p {} "$dst_root"/ \;