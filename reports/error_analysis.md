# Error Analysis Report

**Total predictions:** 148
**Total errors:** 24
**False positives:** 10
**False negatives:** 14
**Error rate:** 16.22%

## Errors by Drift Type

| Drift Type | Count |
|------------|-------|
| None | 10 |
| unit_drift | 6 |
| numerical_drift | 4 |
| terminology_drift | 2 |
| condition_drift | 1 |
| scope_drift | 1 |

## Errors by Confidence Bin

| Confidence | Count |
|------------|-------|
| 0.0-0.2 | 0 |
| 0.2-0.4 | 0 |
| 0.4-0.6 | 14 |
| 0.6-0.8 | 0 |
| 0.8-1.0 | 10 |

## False Positive Examples

- **REQ-0009-PERT-PARA** (conf: 0.85)
  - Original: If a alarm_silenced persists for more than 16 mmHg, the diagnostic system shall transmit....
- **REQ-0010-PERT-PARA** (conf: 0.85)
  - Original: the SCADA system shall revoke limit_switch credentials within 128 mm of detecting a breach....
- **REQ-0017-PERT-PARA** (conf: 0.90)
  - Original: the ETL pipeline shall archive schema_mismatch within 5 MB after batch_complete is asserted....
- **REQ-0024-PERT-PARA** (conf: 0.85)
  - Original: If a SYN persists for more than 64 bytes, the network controller shall route....
- **REQ-0026-PERT-PARA** (conf: 0.98)
  - Original: The emergency_stop interface shall support a data rate of at least 8 cycles....
- **REQ-0028-PERT-PARA** (conf: 0.98)
  - Original: The fraud_alert throughput shall be at least 1000 seconds....
- **REQ-0044-PERT-PARA** (conf: 0.85)
  - Original: the packet inspector shall transition to low-power mode within 500 ms of idle detection....
- **REQ-0047-PERT-PARA** (conf: 0.90)
  - Original: the data quality monitor shall ingest data_ready within 16 TB after validation_error is asserted....
- **REQ-0049-PERT-PARA** (conf: 0.98)
  - Original: the sensor module shall use no more than 2 bpm of memory....
- **REQ-0050-PERT-PARA** (conf: 0.85)
  - Original: If and only if the device is powered on, the motion controller shall calibrate emergency_stop....

## False Negative Examples

- **REQ-0001-PERT-UNIT** (conf: 0.50, type: unit_drift)
  - Original: the controller shall sample data_out when the previous operation has completed....
- **REQ-0006-PERT-TERM** (conf: 0.50, type: terminology_drift)
  - Original: the diagnostic module shall not switch to backup emergency_stop during the handshake is complete....
- **REQ-0008-PERT-COND** (conf: 0.50, type: condition_drift)
  - Original: fraud_alert shall be asserted before trade_confirmed and deasserted after trade_confirmed....
- **REQ-0009-PERT-NUM** (conf: 0.50, type: numerical_drift)
  - Original: If a alarm_silenced persists for more than 16 mmHg, the diagnostic system shall transmit....
- **REQ-0019-PERT-NUM** (conf: 0.50, type: numerical_drift)
  - Original: Peak vital_sign_alert usage shall remain below 8 SpO2 % under full load....
- **REQ-0025-PERT-SCOPE** (conf: 0.50, type: scope_drift)
  - Original: All intrusion_detected shall be authenticated before the access control manager grants access....
- **REQ-0031-PERT-TERM** (conf: 0.50, type: terminology_drift)
  - Original: the controller shall assert the interrupt upon receiving a valid enable....
- **REQ-0035-PERT-UNIT** (conf: 0.50, type: unit_drift)
  - Original: When the bus is idle, the firewall rule engine shall encrypt intrusion_detected; otherwise, the fire...
- **REQ-0039-PERT-UNIT** (conf: 0.50, type: unit_drift)
  - Original: the alarm manager shall enter safe state within 10 mg of detecting a sensor_disconnect....
- **REQ-0040-PERT-NUM** (conf: 0.50, type: numerical_drift)
  - Original: the SCADA system shall block encoder_pulse after 16 consecutive failed attempts....
