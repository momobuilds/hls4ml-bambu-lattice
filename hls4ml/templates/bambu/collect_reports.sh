# Copy final reports (timing, clock, resource usage) to final_reports
mkdir -p final_reports
find HLS_output/Synthesis/vivado_flow -type f -name "*.rpt" -exec cp -p -t final_reports {} +