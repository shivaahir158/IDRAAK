"""Synthetic technical requirement dataset generator."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from idraak.schemas.dataset import DatasetEntry
from idraak.utils.logging import get_logger

logger = get_logger("dataset_generator")

# ── Domain templates ──────────────────────────────────────────────────────────

DOMAINS = {
    "digital_hardware": {
        "actors": ["the controller", "the processor", "the DMA engine", "the arbiter", "the decoder", "the clock manager", "the bus master", "the memory controller", "the register file", "the pipeline stage"],
        "signals": ["ready", "valid", "acknowledge", "request", "grant", "enable", "reset", "clock", "data_out", "interrupt", "chip_select", "write_enable", "read_enable", "busy", "done", "error_flag"],
        "actions": ["assert", "deassert", "latch", "sample", "drive", "propagate", "toggle", "generate", "buffer", "decode"],
        "units": ["clock cycles", "ns", "ps", "MHz", "GHz", "bits", "bytes"],
    },
    "embedded_systems": {
        "actors": ["the firmware", "the RTOS scheduler", "the watchdog timer", "the bootloader", "the interrupt handler", "the sensor driver", "the power manager", "the device driver", "the task scheduler", "the flash controller"],
        "signals": ["heartbeat", "timeout", "wake_event", "sleep_mode", "low_power_state", "calibration_flag", "overflow", "underflow"],
        "actions": ["initialize", "configure", "poll", "reset", "enter sleep mode", "trigger", "schedule", "halt", "resume", "flush"],
        "units": ["ms", "us", "KB", "MHz", "mA", "mW", "bytes"],
    },
    "software_systems": {
        "actors": ["the application", "the service", "the middleware", "the API gateway", "the database connector", "the cache layer", "the message queue", "the scheduler", "the logging module", "the authentication module"],
        "signals": ["request", "response", "callback", "event", "exception", "timeout_signal", "heartbeat"],
        "actions": ["process", "validate", "return", "log", "retry", "queue", "invoke", "authenticate", "serialize", "deserialize"],
        "units": ["ms", "seconds", "requests per second", "MB", "GB", "connections"],
    },
    "networking": {
        "actors": ["the router", "the switch", "the firewall", "the load balancer", "the DNS resolver", "the proxy server", "the network controller", "the packet inspector", "the DHCP server", "the NAT gateway"],
        "signals": ["SYN", "ACK", "FIN", "RST", "packet", "frame", "segment"],
        "actions": ["forward", "drop", "queue", "route", "encapsulate", "decrypt", "filter", "inspect", "throttle", "retransmit"],
        "units": ["Mbps", "Gbps", "ms", "packets per second", "hops", "bytes", "TTL"],
    },
    "cybersecurity": {
        "actors": ["the intrusion detection system", "the authentication server", "the encryption module", "the access control manager", "the certificate authority", "the security monitor", "the key manager", "the audit logger", "the firewall rule engine", "the vulnerability scanner"],
        "signals": ["alert", "violation", "intrusion_detected", "access_denied", "token_expired", "certificate_invalid"],
        "actions": ["block", "encrypt", "decrypt", "authenticate", "authorize", "revoke", "quarantine", "scan", "audit", "escalate"],
        "units": ["bits", "bytes", "attempts", "seconds", "days"],
    },
    "safety_critical": {
        "actors": ["the safety controller", "the emergency shutdown system", "the fault monitor", "the redundancy manager", "the watchdog", "the failsafe unit", "the diagnostic module", "the backup controller", "the protection relay", "the alarm system"],
        "signals": ["fault", "alarm", "emergency_stop", "safe_state", "degraded_mode", "recovery_signal", "warning"],
        "actions": ["shut down", "engage failsafe", "activate alarm", "enter safe state", "isolate", "switch to backup", "log fault", "notify operator", "disable", "verify integrity"],
        "units": ["ms", "seconds", "cycles", "faults per hour", "probability"],
    },
    "data_processing": {
        "actors": ["the ETL pipeline", "the data validator", "the transformation engine", "the aggregation service", "the data lake ingester", "the stream processor", "the batch scheduler", "the schema validator", "the deduplication engine", "the data quality monitor"],
        "signals": ["batch_complete", "validation_error", "schema_mismatch", "data_ready"],
        "actions": ["ingest", "transform", "validate", "aggregate", "partition", "deduplicate", "enrich", "normalize", "archive", "purge"],
        "units": ["records per second", "MB", "GB", "TB", "rows", "partitions"],
    },
    "financial_systems": {
        "actors": ["the transaction processor", "the risk engine", "the settlement system", "the fraud detector", "the compliance checker", "the ledger service", "the payment gateway", "the reconciliation engine", "the margin calculator", "the audit trail"],
        "signals": ["trade_confirmed", "settlement_failed", "fraud_alert", "margin_call", "compliance_violation"],
        "actions": ["process", "settle", "validate", "reconcile", "calculate", "authorize", "reverse", "hold", "release", "report"],
        "units": ["ms", "transactions per second", "USD", "basis points", "seconds"],
    },
    "healthcare_devices": {
        "actors": ["the patient monitor", "the infusion pump controller", "the ventilator", "the diagnostic system", "the alarm manager", "the dosage calculator", "the sensor module", "the data recorder", "the display controller", "the communication interface"],
        "signals": ["vital_sign_alert", "dose_limit_exceeded", "sensor_disconnect", "battery_low", "alarm_silenced"],
        "actions": ["monitor", "administer", "alert", "calculate dosage", "record", "display", "calibrate", "silence alarm", "escalate", "transmit"],
        "units": ["mL/hr", "bpm", "mmHg", "SpO2 %", "seconds", "mg", "mcg/kg/min"],
    },
    "industrial_automation": {
        "actors": ["the PLC", "the SCADA system", "the HMI", "the motion controller", "the robot controller", "the conveyor system", "the temperature controller", "the pressure regulator", "the flow meter", "the vision system"],
        "signals": ["limit_switch", "proximity_sensor", "encoder_pulse", "emergency_stop", "cycle_complete", "home_position"],
        "actions": ["actuate", "regulate", "position", "control", "measure", "calibrate", "sequence", "interlock", "ramp", "dwell"],
        "units": ["rpm", "mm", "degrees", "bar", "PSI", "liters per minute", "ms", "degrees Celsius"],
    },
}

# ── Category templates ────────────────────────────────────────────────────────

CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "functional_behavior": [
        "{actor} shall {action} {signal} when {condition}.",
        "{actor} shall {action} the {signal} upon receiving a valid {signal2}.",
        "{actor} shall {action} the {signal} and update the {signal2} register.",
        "When {condition}, {actor} shall {action} {signal} to the output port.",
    ],
    "timing_constraint": [
        "{actor} shall {action} {signal} within {value} {unit} after {signal2} is asserted.",
        "The {signal} shall be stable for at least {value} {unit} before the {signal2} edge.",
        "{actor} shall complete the {action} operation within {value} {unit}.",
        "The latency from {signal} assertion to {signal2} response shall not exceed {value} {unit}.",
    ],
    "numerical_constraint": [
        "The {signal} width shall be exactly {value} {unit}.",
        "{actor} shall support at least {value} {unit} of {signal} capacity.",
        "The maximum {signal} frequency shall be {value} {unit}.",
        "The {signal} value shall be between {value} and {value2} {unit}.",
    ],
    "resource_constraint": [
        "{actor} shall use no more than {value} {unit} of memory.",
        "The total {signal} consumption shall not exceed {value} {unit}.",
        "{actor} shall operate with at most {value} concurrent {signal} connections.",
        "Peak {signal} usage shall remain below {value} {unit} under full load.",
    ],
    "conditional_behavior": [
        "If {condition} and {condition2}, then {actor} shall {action} {signal}.",
        "{actor} shall {action} {signal} only if {condition}.",
        "If and only if {condition}, {actor} shall {action} {signal}.",
        "When {condition}, {actor} shall {action} {signal}; otherwise, {actor} shall {action} {signal2}.",
    ],
    "negation": [
        "{actor} shall not {action} {signal} during {condition}.",
        "{actor} shall not {action} {signal} unless {condition}.",
        "Under no circumstances shall {actor} {action} {signal} while {condition2}.",
        "{actor} shall not modify the {signal} register when {condition}.",
    ],
    "exception_handling": [
        "{actor} shall {action} {signal} unless {exception}.",
        "If {condition}, {actor} shall {action} {signal}, except when {exception}.",
        "{actor} shall {action} {signal} in all cases except when {exception}.",
        "Unless {exception}, {actor} shall {action} {signal} within {value} {unit}.",
    ],
    "sequence_ordering": [
        "{actor} shall {action} {signal} before {action2} {signal2}.",
        "{actor} shall first {action} {signal} and then {action2} {signal2}.",
        "{signal} shall be asserted before {signal2} and deasserted after {signal2}.",
        "The {action} of {signal} shall always precede the {action2} of {signal2}.",
    ],
    "safety_requirement": [
        "{actor} shall enter safe state within {value} {unit} of detecting a {signal}.",
        "If a {signal} persists for more than {value} {unit}, {actor} shall {action}.",
        "{actor} shall maintain {signal} redundancy at all times during safety-critical operations.",
        "Upon detecting {signal}, {actor} shall immediately {action} and activate the {signal2}.",
    ],
    "security_requirement": [
        "{actor} shall {action} all {signal} data using AES-{value}-bit encryption.",
        "{actor} shall block {signal} after {value} consecutive failed attempts.",
        "All {signal} shall be authenticated before {actor} grants access.",
        "{actor} shall revoke {signal} credentials within {value} {unit} of detecting a breach.",
    ],
    "interface_requirement": [
        "{actor} shall communicate with the {signal} interface using the {signal2} protocol.",
        "The {signal} interface shall support a data rate of at least {value} {unit}.",
        "{actor} shall acknowledge each {signal} transaction within {value} {unit}.",
        "The {signal} bus width shall be {value} {unit}.",
    ],
    "reliability_requirement": [
        "{actor} shall achieve a mean time between failures of at least {value} hours.",
        "The {signal} subsystem shall have an availability of at least {value}%.",
        "{actor} shall recover from a {signal} failure within {value} {unit}.",
        "The probability of {signal} failure shall be less than {value} per hour.",
    ],
    "performance_requirement": [
        "{actor} shall process at least {value} {unit} under nominal conditions.",
        "The {signal} throughput shall be at least {value} {unit}.",
        "{actor} shall achieve a response time of no more than {value} {unit} for {signal} requests.",
        "The {signal} processing latency shall remain below {value} {unit} at the 99th percentile.",
    ],
    "power_requirement": [
        "{actor} shall consume no more than {value} {unit} in active mode.",
        "In sleep mode, {actor} shall draw less than {value} {unit}.",
        "The total {signal} power budget shall not exceed {value} {unit}.",
        "{actor} shall transition to low-power mode within {value} {unit} of idle detection.",
    ],
    "compliance_requirement": [
        "{actor} shall comply with {standard} requirements for {signal}.",
        "All {signal} operations shall conform to {standard} section {value}.",
        "{actor} shall meet the {standard} certification requirements for {signal} handling.",
        "The {signal} interface shall be compliant with {standard} revision {value}.",
    ],
}

STANDARDS = [
    "ISO 26262", "IEC 61508", "DO-178C", "IEC 62304", "IEEE 802.3",
    "MISRA C", "AUTOSAR", "ISO 27001", "NIST 800-53", "PCI DSS",
    "HIPAA", "FDA 21 CFR Part 11", "ISO 13485", "IEC 60601",
]

CONDITIONS = [
    "the system is in active mode",
    "the input is valid",
    "the bus is idle",
    "the clock is stable",
    "the device is powered on",
    "the handshake is complete",
    "the configuration register is set",
    "the interrupt is enabled",
    "the buffer is not full",
    "the channel is available",
    "the system has been initialized",
    "the previous operation has completed",
    "the error flag is clear",
    "the watchdog has not timed out",
    "the temperature is within the operating range",
]

EXCEPTIONS = [
    "a critical fault is detected",
    "the system is in debug mode",
    "the emergency stop is activated",
    "a power failure occurs",
    "the reset signal is active",
    "the system is in maintenance mode",
    "an overflow condition exists",
    "the calibration period is active",
    "the hardware self-test is running",
    "a higher-priority interrupt is pending",
]


class DatasetGenerator:
    """Generates synthetic technical requirement datasets."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.counter = 0

    def generate(self, n: int = 300) -> list[DatasetEntry]:
        """Generate n diverse technical requirements."""
        entries: list[DatasetEntry] = []
        domains = list(DOMAINS.keys())
        categories = list(CATEGORY_TEMPLATES.keys())

        for i in range(n):
            domain = domains[i % len(domains)]
            category = categories[i % len(categories)]
            entry = self._generate_one(domain, category)
            entries.append(entry)

        logger.info(f"Generated {len(entries)} requirements across {len(domains)} domains")
        return entries

    def _generate_one(self, domain: str, category: str) -> DatasetEntry:
        self.counter += 1
        req_id = f"REQ-{self.counter:04d}"
        d = DOMAINS[domain]
        templates = CATEGORY_TEMPLATES[category]
        template = self.rng.choice(templates)

        actor = self.rng.choice(d["actors"])
        signal = self.rng.choice(d["signals"])
        signal2 = self.rng.choice([s for s in d["signals"] if s != signal] or d["signals"])
        action = self.rng.choice(d["actions"])
        action2 = self.rng.choice([a for a in d["actions"] if a != action] or d["actions"])
        condition = self.rng.choice(CONDITIONS)
        condition2 = self.rng.choice([c for c in CONDITIONS if c != condition])
        exception = self.rng.choice(EXCEPTIONS)
        unit = self.rng.choice(d["units"])
        value = self.rng.choice([1, 2, 3, 4, 5, 8, 10, 16, 32, 50, 64, 100, 128, 256, 500, 1000])
        value2 = value * self.rng.choice([2, 4, 5, 10])
        standard = self.rng.choice(STANDARDS)

        text = template.format(
            actor=actor, signal=signal, signal2=signal2,
            action=action, action2=action2,
            condition=condition, condition2=condition2,
            exception=exception, unit=unit,
            value=value, value2=value2, standard=standard,
        )

        difficulty = self._estimate_difficulty(text, category)
        critical_attrs = self._extract_critical_attributes(category)

        return DatasetEntry(
            requirement_id=req_id,
            domain=domain,
            category=category,
            original_text=text,
            critical_attributes=critical_attrs,
            difficulty=difficulty,
            source="synthetic",
            complexity_score=self._compute_complexity(text),
        )

    def _estimate_difficulty(self, text: str, category: str) -> str:
        score = 0
        if len(text.split()) > 20:
            score += 1
        if any(w in text.lower() for w in ["unless", "except", "only if", "if and only if"]):
            score += 1
        if category in ("negation", "exception_handling", "conditional_behavior"):
            score += 1
        if text.count(",") > 2:
            score += 1
        if score >= 3:
            return "hard"
        if score >= 1:
            return "medium"
        return "easy"

    def _extract_critical_attributes(self, category: str) -> list[str]:
        mapping = {
            "timing_constraint": ["timing.value", "timing.unit", "timing.relation"],
            "numerical_constraint": ["numerical.value", "numerical.unit", "numerical.operator"],
            "negation": ["polarity", "modality"],
            "conditional_behavior": ["conditions", "logical_operator"],
            "exception_handling": ["exceptions", "conditions"],
            "sequence_ordering": ["ordering", "temporal_relation"],
            "safety_requirement": ["safety_constraints", "timing.value"],
            "security_requirement": ["security_constraints", "numerical.value"],
            "functional_behavior": ["action", "object", "modality"],
            "resource_constraint": ["numerical.value", "numerical.unit", "numerical.operator"],
            "interface_requirement": ["interface_entities", "numerical.value"],
            "reliability_requirement": ["numerical.value", "numerical.unit"],
            "performance_requirement": ["numerical.value", "numerical.unit", "numerical.operator"],
            "power_requirement": ["numerical.value", "numerical.unit"],
            "compliance_requirement": ["qualifiers"],
        }
        return mapping.get(category, ["action", "modality"])

    def _compute_complexity(self, text: str) -> float:
        tokens = len(text.split())
        clauses = text.count(",") + text.count(";") + 1
        conditions = sum(1 for w in ["if", "when", "unless", "only", "except", "before", "after", "within"]
                         if w in text.lower().split())
        numbers = sum(1 for w in text.split() if w.replace(".", "").replace("-", "").isdigit())
        return min(1.0, (tokens / 40 + clauses / 5 + conditions / 3 + numbers / 3) / 4)

    def save(self, entries: list[DatasetEntry], path: str | Path) -> None:
        """Save dataset to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for e in entries:
                f.write(e.model_dump_json() + "\n")
        logger.info(f"Saved {len(entries)} entries to {path}")

    @staticmethod
    def load(path: str | Path) -> list[DatasetEntry]:
        """Load dataset from JSONL."""
        entries = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    entries.append(DatasetEntry.model_validate_json(line))
        return entries
