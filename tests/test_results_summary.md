# Test Results Summary for Individual Permutation Group Datasets

## Test Status by Group Type

### ✅ Passing Tests (Correct Data)
- **Symmetric Groups (S3-S9)**: All tests pass
- **Alternating Groups (A3-A9)**: All tests pass  
- **Cyclic Groups (C3-C30)**: All tests pass (1 skipped)
- **Dihedral Groups (D3-D20)**: All tests pass
- **Klein Four Group (V4)**: All tests pass

### ❌ Failing Tests (Incorrect Data)
- **Quaternion Groups (Q8, Q16, Q32)**:
  - Q8 uses IDs 0-15 instead of 0-7 (should have order 8)
  - Q16 and Q32 may have similar issues
  
- **Elementary Abelian Groups**:
  - Many tests skipped due to missing datasets
  - Existing datasets pass their tests
  
- **Frobenius Groups (F20, F21)**:
  - F20 shows degree 5 instead of 20
  - F21 shows degree 7 instead of 21
  
- **PSL Groups (PSL(2,5), PSL(2,7))**:
  - ID bounds exceed expected group orders
  
- **Mathieu Groups (M11, M12)**:
  - M11 has IDs in millions (e.g., 20913103) when order should be 7920
  - M12 likely has similar issues
  - Multiple test failures including duplicates and poor target coverage

## Root Cause Analysis

The failing groups appear to have been generated incorrectly when splitting from supersets. The issues include:
1. Wrong ID ranges (quaternion, PSL, Mathieu)
2. Wrong degree values (Frobenius)
3. Potential data corruption (Mathieu groups with massive IDs)

## Recommendation

The following groups need to be regenerated from scratch with correct parameters:
- Quaternion groups (Q8, Q16, Q32)
- Frobenius groups (F20, F21)
- PSL groups (PSL(2,5), PSL(2,7))
- Mathieu groups (M11, M12)
- Elementary Abelian groups (verify all exist with correct parameters)