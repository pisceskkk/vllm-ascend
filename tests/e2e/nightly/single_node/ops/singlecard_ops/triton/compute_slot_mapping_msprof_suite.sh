#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
entry=${script_dir}/compute_slot_mapping_profile_entry.py
root=${SLOT_MAP_PROFILE_ROOT:?SLOT_MAP_PROFILE_ROOT is required}
launch_count=${SLOT_MAP_PROFILE_REPEATS:-20}
warmup=${SLOT_MAP_PROFILE_WARMUP:-20}
use_parallel_kernel=${SLOT_MAP_USE_PARALLEL_KERNEL:-0}
case_set=${SLOT_MAP_PROFILE_CASE_SET:-all}
mkdir -p "${root}"
summary="${root}/summary.tsv"
printf 'case_id\trepeats\tmedian_us\tmean_us\tmin_us\tmax_us\tcv\n' > "${summary}"

cases=(
  profile-1r-4096-s64
  profile-1r-4096-s2048
  profile-1r-4096-s8192
  profile-1r-4096-s65536
  profile-1r-4096-s131072
  profile-2r-4096-s8192
  decode-1r-1-max4096
  decode-64r-64-max4096
  tail-1r-1023
  boundary-1r-1025
  prefill-8r-8192
  hybrid-1r-4096-p128-l32
  padding-1r-4-max8192
)
if [[ "${case_set}" == affected ]]; then
  cases=(
    profile-1r-4096-s64
    profile-1r-4096-s2048
    profile-1r-4096-s8192
    profile-1r-4096-s65536
    profile-1r-4096-s131072
    hybrid-1r-4096-p128-l32
  )
fi

for case_id in "${cases[@]}"; do
  case_root="${root}/${case_id}"
  kernel_name=_compute_slot_mapping_kernel
  if [[ "${use_parallel_kernel}" == 1 ]]; then
    case "${case_id}" in
      profile-* | hybrid-1r-4096-p128-l32)
        kernel_name=_compute_slot_mapping_parallel_kernel
        ;;
    esac
  fi
  SLOT_MAP_CASE="${case_id}" SLOT_MAP_LAUNCHES=200 \
    msprof op \
      --application="${entry}" \
      --output="${case_root}" \
      --kernel-name="${kernel_name}" \
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
