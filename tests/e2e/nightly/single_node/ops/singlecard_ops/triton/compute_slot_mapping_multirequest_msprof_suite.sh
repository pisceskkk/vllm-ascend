#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
entry=${script_dir}/compute_slot_mapping_multirequest_profile_entry.py
root=${SLOT_MAP_PROFILE_ROOT:?SLOT_MAP_PROFILE_ROOT is required}
launch_count=${SLOT_MAP_PROFILE_REPEATS:-20}
warmup=${SLOT_MAP_PROFILE_WARMUP:-20}
metrics=${SLOT_MAP_AIC_METRICS:-BasicInfo}
mkdir -p "${root}"
summary="${root}/summary.tsv"
printf 'case_id\tfirst_mode\toriginal_median_us\tcurrent_median_us\tspeedup_pct\toriginal_cv\tcurrent_cv\n' > "${summary}"

cases=(
  profile-2r-4096-s8192
  profile-2r-uneven-s8192
  profile-mr-4x1024-s8192
  profile-mr-8x512-s8192
  profile-mr-16x256-s8192
  profile-mr-32x128-s8192
  profile-mr-64x64-s8192
  profile-mr-8-uneven-s8192
  profile-mr-32-uneven-s8192
  prefill-8r-8192
  decode-64r-64-max4096
)

summarize_profile() {
  local case_id=$1
  local mode=$2
  local case_root=$3
  find "${case_root}" -name 'OpBasicInfo_*.csv' \
    -exec awk -F, 'FNR == 2 {print $3}' {} + \
    | sort -n > "${root}/${case_id}.${mode}.durations-us.txt"
  awk '
    {a[NR]=$1; sum+=$1; sumsq+=$1*$1}
    END {
      n=NR;
      median=(n%2 ? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2);
      mean=sum/n;
      cv=sqrt(sumsq/n-mean*mean)/mean;
      printf "%.6f\t%.6f\n", median,cv
    }
  ' "${root}/${case_id}.${mode}.durations-us.txt" > "${root}/${case_id}.${mode}.stats.tsv"
}

run_profile() {
  local case_id=$1
  local mode=$2
  local kernel_name=_compute_slot_mapping_kernel
  local case_root="${root}/${case_id}/${mode}"
  if [[ "${mode}" == current ]]; then
    case "${case_id}" in
      profile-2r-4096-s8192 | profile-2r-uneven-s8192)
        kernel_name=_compute_slot_mapping_parallel_kernel
        ;;
      profile-mr-8x512-s8192 | profile-mr-16x256-s8192 | profile-mr-32x128-s8192 | \
        profile-mr-64x64-s8192 | profile-mr-8-uneven-s8192 | \
        profile-mr-32-uneven-s8192 | decode-64r-64-max4096)
        kernel_name=_compute_slot_mapping_adaptive_kernel
        ;;
    esac
  fi
  mkdir -p "${case_root}"
  SLOT_MAP_CASE="${case_id}" SLOT_MAP_LAUNCH_MODE="${mode}" SLOT_MAP_LAUNCHES=200 \
    msprof op \
      --application="${entry}" \
      --output="${case_root}" \
      --kernel-name="${kernel_name}" \
      --launch-count="${launch_count}" \
      --warm-up="${warmup}" \
      --aic-metrics="${metrics}" \
      --replay-mode=kernel \
      > "${root}/${case_id}.${mode}.log" 2>&1
  summarize_profile "${case_id}" "${mode}" "${case_root}"
}

case_index=0
for case_id in "${cases[@]}"; do
  if ((case_index % 2 == 0)); then
    modes=(original current)
  else
    modes=(current original)
  fi
  first_mode=${modes[0]}
  for mode in "${modes[@]}"; do
    run_profile "${case_id}" "${mode}"
  done
  read -r original_median original_cv < "${root}/${case_id}.original.stats.tsv"
  read -r current_median current_cv < "${root}/${case_id}.current.stats.tsv"
  awk -v case_id="${case_id}" -v first_mode="${first_mode}" \
    -v original="${original_median}" -v current="${current_median}" \
    -v original_cv="${original_cv}" -v current_cv="${current_cv}" '
      BEGIN {
        speedup=(original/current-1)*100;
        printf "%s\t%s\t%.6f\t%.6f\t%.3f\t%.6f\t%.6f\n", \
          case_id,first_mode,original,current,speedup,original_cv,current_cv
      }
    ' >> "${summary}"
  case_index=$((case_index + 1))
done
