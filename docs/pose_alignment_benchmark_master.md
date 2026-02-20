# HEAL Benchmark Master Table (Full Sweep, Single Comm=0)

## Best Strategy per Method (by mean AP50)
- Single rows use the same coop checkpoints with comm_range_override=0 and fusion_method=intermediate.
- Strategy selection is done separately for noise sweep and dropout sweep.

### camera
| method | noise_strategy | noise_ap50 | noise_success@5m | noise_rel_trans_m | dropout_strategy | dropout_ap50 | dropout_success@5m | single_noise_ap50 | single_dropout_ap50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_comm0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.052 | 0.040 |
| baseline | bounds | 0.123 | 0.462 | 6.857 | bounds | 0.098 | 0.369 | n/a | n/a |
| oracle | bounds | 0.223 | 1.000 | 0.000 | bounds | 0.223 | 1.000 | n/a | n/a |
| v2xregpp | best | 0.124 | 0.464 | 6.848 | stable | 0.119 | 0.074 | n/a | n/a |
| freealign | best | 0.114 | 0.429 | 11.425 | best | 0.109 | 0.350 | n/a | n/a |
| vips | best | 0.114 | 0.374 | 19.642 | best | 0.109 | 0.308 | n/a | n/a |
| cbm | best | 0.098 | 0.091 | 55.764 | best | 0.095 | 0.069 | n/a | n/a |

### lidar
| method | noise_strategy | noise_ap50 | noise_success@5m | noise_rel_trans_m | dropout_strategy | dropout_ap50 | dropout_success@5m | single_noise_ap50 | single_dropout_ap50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_comm0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.500 | 0.388 |
| baseline | bounds | 0.300 | 0.462 | 6.857 | bounds | 0.229 | 0.369 | n/a | n/a |
| oracle | bounds | 0.431 | 1.000 | 0.000 | bounds | 0.431 | 1.000 | n/a | n/a |
| v2xregpp | best | 0.394 | 0.688 | 5.170 | best | 0.391 | 0.561 | n/a | n/a |
| freealign | best | 0.382 | 0.633 | 5.335 | best | 0.381 | 0.509 | n/a | n/a |
| vips | best | 0.298 | 0.387 | 19.249 | best | 0.314 | 0.319 | n/a | n/a |
| cbm | stable | 0.379 | 0.009 | 118.354 | stable | 0.342 | 0.005 | n/a | n/a |

