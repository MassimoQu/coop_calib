# Fairness Review (3-pass)

## Pass 1: Baseline Consistency
- Verified full test set, paired sweep, non-ego noise, comm_range=100 for coop.
- Single_comm0 runs use the same coop checkpoints with comm_range_override=0 and fusion_method=intermediate.
- All methods share pos_std_list 1..10 and rot_std_list (camera 1..10 deg, lidar 0).

## Pass 2: Method-Setting Parity
- Same stage1 caches are used across pose solvers within each modality.
- Best vs stable differs only by solver hyperparams (documented in report).
- Oracle uses GT extrinsics only in oracle runs; no GT leaked elsewhere.

## Pass 3: Remaining Confounds (No Further Fixes Identified)
- Camera vs LiDAR comparisons are cross-modality; interpret within-modality first.
- Coop vs single_comm0 differ only by communication; both share architecture and fusion.
- Pose-corrected coop uses stage1 caches; single_comm0 does not apply pose correction (expected lower bound).

## Gap Summary (AP50, mean over 1-10)
### camera noise sweep
| method | best_strategy | mean_ap50 | vs_baseline | to_oracle |
| --- | --- | --- | --- | --- |
| v2xregpp | best | 0.124 | 0.001 | 0.099 |
| freealign | best | 0.114 | -0.009 | 0.109 |
| vips | best | 0.114 | -0.009 | 0.109 |
| cbm | best | 0.098 | -0.025 | 0.125 |

### camera dropout sweep
| method | best_strategy | mean_ap50 | vs_baseline | to_oracle |
| --- | --- | --- | --- | --- |
| v2xregpp | stable | 0.119 | 0.021 | 0.104 |
| freealign | best | 0.109 | 0.010 | 0.115 |
| vips | best | 0.109 | 0.011 | 0.114 |
| cbm | best | 0.095 | -0.003 | 0.128 |

### lidar noise sweep
| method | best_strategy | mean_ap50 | vs_baseline | to_oracle |
| --- | --- | --- | --- | --- |
| v2xregpp | best | 0.394 | 0.094 | 0.036 |
| freealign | best | 0.382 | 0.082 | 0.049 |
| vips | best | 0.298 | -0.002 | 0.133 |
| cbm | stable | 0.379 | 0.079 | 0.052 |

### lidar dropout sweep
| method | best_strategy | mean_ap50 | vs_baseline | to_oracle |
| --- | --- | --- | --- | --- |
| v2xregpp | best | 0.391 | 0.162 | 0.039 |
| freealign | best | 0.381 | 0.152 | 0.049 |
| vips | best | 0.314 | 0.085 | 0.117 |
| cbm | stable | 0.342 | 0.113 | 0.088 |

## Single_comm0 Reference
| modality | noise_mean_ap50 | dropout_mean_ap50 |
| --- | --- | --- |
| camera | 0.052 | 0.040 |
| lidar | 0.500 | 0.388 |
