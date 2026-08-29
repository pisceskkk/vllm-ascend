#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
entry=${script_dir}/compute_slot_mapping_multigroup_profile_entry.py
root=${SLOT_MAP_PROFILE_ROOT:?SLOT_MAP_PROFILE_ROOT is required}
launch_count=${SLOT_MAP_PROFILE_REPEATS:-20}
warmup=${SLOT_MAP_PROFILE_WARMUP:-20}
mkdir -p "${root}"
summary="${root}/summary.tsv"
printf 'case_id\trepeats\tmedian_us\tmean_us\tmin_us\tmax_us\tcv\n' > "${summary}"

cases=(
  multigroup-profile-6x4096
  multigroup-general-2x2048-padding
)

for case_id in "${cases[@]}"; do
  case_root="${root}/${case_id}"
  SLOT_MAP_MULTIGROUP_CASE="${case_id}" SLOT_MAP_LAUNCHES=200 \
    msprof op \
      --application="${entry}" \
      --output="${case_root}" \
      --kernel-name=_compute_slot_mapping_fused_groups_kernel \
      --launch-count="${launch_count}" \
      --warm-up="${warmup}" \
      --aic-metrics=BasicInfo \
      --replay-mode=kernel \
      > "${case_root}.log" 2>&1
  find "${case_root}" -name 'OpBasicInfo_*.csv' \
    -exec awk -F, 'FNR == 2 {print $3}' {} + \
    | sort -n > "${case_root}.durations-us.txt"
  awk -v case_id="${case_id}" '
    {a[NR]=$1; sum+=$1; sumsq+=$1*$1}
    END {
      n=NR;
      median=(n%2 ? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2);
      mean=sum/n;
      cv=sqrt(sumsq/n-mean*mean)/mean;
      printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", case_id,n,median,mean,a[1],a[n],cv
    }
  ' "${case_root}.durations-us.txt" >> "${summary}"
done
