# IDRAAK Annotation Guidelines

## Task

For each requirement pair, determine whether semantic drift has occurred between the original English requirement and the translated/paraphrased version.

## Labels

### No Drift (0)
The translated version preserves the exact same meaning as the original. Acceptable variations include:
- Different sentence structure with same meaning
- Synonym usage that preserves technical accuracy
- Unit conversions with equivalent physical values (e.g., 1000 ms = 1 s)
- Reordered conjunctions (A and B = B and A)
- Voice changes (active to passive) that preserve meaning

### Drift (1)
The translated version has a different meaning from the original. This includes:

#### Minor Drift (severity: low)
- Slight rephrasing that introduces ambiguity but likely preserves intent
- Terminology variation that could be interpreted differently in context

#### Moderate Drift (severity: medium)
- Changed scope (all -> most)
- Added or removed non-critical qualifiers
- Terminology replacement with a related but different term

#### Major Drift (severity: high)
- Changed numerical values
- Changed units without conversion
- Changed modality (shall -> should)
- Changed logical operators (and -> or)
- Changed temporal relations (before -> after)
- Reversed threshold direction (at least -> at most)
- Removed exception clauses

#### Critical Drift (severity: critical)
- Inverted polarity (shall -> shall not)
- Changed modality from mandatory to forbidden
- Changed safety-critical constraints
- Omitted safety or security requirements

## Examples

### Semantic Equivalence (No Drift)
- Original: "The controller shall assert ready within 3 clock cycles."
- Equivalent: "Ready must be asserted by the controller within 3 clock cycles."

### Minor Drift
- Original: "The system shall process all requests."
- Drifted: "The system shall handle all requests." (terminology variation)

### Major Drift
- Original: "The latency shall not exceed 10 ms."
- Drifted: "The latency shall not exceed 10 seconds." (unit drift)

### Critical Drift
- Original: "The system shall encrypt all data."
- Drifted: "The system shall not encrypt all data." (polarity inversion)

## Rating Scale

For each pair, provide:
1. **Drift present?** (Yes/No)
2. **Drift type** (select from: numerical, unit, polarity, modality, condition, temporal, threshold, entity, relation, exception, omission, addition, terminology, scope, reference)
3. **Severity** (none/low/medium/high/critical)
4. **Is the system explanation correct?** (Yes/Partially/No)
5. **Is the evidence sufficient?** (Yes/No)
6. **Would human review be required?** (Yes/No)
