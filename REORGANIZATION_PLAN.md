# Documentation Reorganization Plan

**Date**: 2025-12-26
**Status**: Ready for implementation
**Purpose**: Improve documentation structure for lab members

---

## 📊 Current State

**19 markdown files** in project root - too many, unclear hierarchy

## 🎯 Proposed Structure

```
xRL_pilot/
├── README.md                    # ⭐ Main entry point
├── PROJECT_OVERVIEW.md          # ⭐ Research overview (Korean)
├── CLAUDE.md                    # Research plan (English)
│
├── docs/                        # Core documentation
│   ├── AIRL_DESIGN.md
│   ├── IMPLEMENTATION_NOTES.md
│   └── IMPLEMENTATION_SUMMARY.md
│
├── progress/                    # Progress tracking
│   ├── PHASE2_PROGRESS.md
│   └── DOCUMENTATION_QUALITY_REVIEW.md
│
└── archive/                     # Outdated/superseded documents
    ├── IMPLEMENTATION_STATUS.md       # → Superseded by PHASE2_PROGRESS.md
    ├── PLANNING_DEPTH_PRINCIPLES.md   # → Incorporated into AIRL_DESIGN.md
    ├── RESPONSE_TO_FEEDBACK.md        # → Historical
    ├── DEPTH_VARIABLE_VERIFICATION.md # → Validation complete
    ├── GYMNASIUM_AND_AIRL_GUIDE.md    # → Superseded by IMPLEMENTATION_NOTES.md
    ├── PHASE2_VALIDATION_CHECKLIST.md # → All checkpoints passed
    ├── DEPTH_INTEGRATION_DETAILED.md  # → Option A selected
    ├── OPTION_A_VS_B.md               # → Decision made
    ├── OPTION_DIFFERENCE_SIMPLE.md    # → Decision made
    ├── OPTION_A_DEPTH_H_EXPLAINED.md  # → Decision made
    ├── RESEARCH_DISCUSSION.md         # → Incorporated into PROJECT_OVERVIEW.md
    ├── AIRL_COMPLETE_GUIDE.md         # → Consolidated into AIRL_DESIGN.md
    ├── PROJECT_SUMMARY.md             # → Phase 1 summary
    └── FOLDER_STRUCTURE.md            # → Phase 1 structure
```

---

## 📁 File Classification

### ⭐ Root (Essential - 3 files)

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Project entry point | ✅ Updated |
| `PROJECT_OVERVIEW.md` | Research overview (Korean) | ✅ New |
| `CLAUDE.md` | Full research plan (English) | ✅ Keep |

### 📖 docs/ (Core Documentation - 3 files)

| File | Purpose | Status |
|------|---------|--------|
| `AIRL_DESIGN.md` | Design document | ✅ Updated |
| `IMPLEMENTATION_NOTES.md` | Technical details | ✅ Current |
| `IMPLEMENTATION_SUMMARY.md` | Implementation summary | ✅ Current |

### 📊 progress/ (Progress Tracking - 2 files)

| File | Purpose | Status |
|------|---------|--------|
| `PHASE2_PROGRESS.md` | Current status | ✅ Current |
| `DOCUMENTATION_QUALITY_REVIEW.md` | Quality check | ✅ Reference |

### 📦 archive/ (Historical/Outdated - 13 files)

| File | Reason for Archive |
|------|-------------------|
| `IMPLEMENTATION_STATUS.md` | Superseded by PHASE2_PROGRESS.md |
| `PLANNING_DEPTH_PRINCIPLES.md` | Incorporated into AIRL_DESIGN.md |
| `RESPONSE_TO_FEEDBACK.md` | Historical discussion |
| `DEPTH_VARIABLE_VERIFICATION.md` | Validation complete |
| `GYMNASIUM_AND_AIRL_GUIDE.md` | Superseded by IMPLEMENTATION_NOTES.md |
| `PHASE2_VALIDATION_CHECKLIST.md` | All checkpoints passed |
| `DEPTH_INTEGRATION_DETAILED.md` | Option A selected |
| `OPTION_A_VS_B.md` | Decision made (Option A) |
| `OPTION_DIFFERENCE_SIMPLE.md` | Decision made |
| `OPTION_A_DEPTH_H_EXPLAINED.md` | Decision made |
| `RESEARCH_DISCUSSION.md` | Incorporated into PROJECT_OVERVIEW.md |
| `AIRL_COMPLETE_GUIDE.md` | Consolidated into AIRL_DESIGN.md |
| `PROJECT_SUMMARY.md` | Phase 1 summary (historical) |
| `FOLDER_STRUCTURE.md` | Phase 1 structure (historical) |

---

## 🔄 Migration Commands

**WARNING**: Review changes before executing!

```bash
cd /Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot

# Create folders
mkdir -p docs progress archive

# Move to docs/
mv AIRL_DESIGN.md docs/
mv IMPLEMENTATION_NOTES.md docs/
mv IMPLEMENTATION_SUMMARY.md docs/

# Move to progress/
mv PHASE2_PROGRESS.md progress/
mv DOCUMENTATION_QUALITY_REVIEW.md progress/

# Move to archive/
mv IMPLEMENTATION_STATUS.md archive/
mv PLANNING_DEPTH_PRINCIPLES.md archive/
mv RESPONSE_TO_FEEDBACK.md archive/
mv DEPTH_VARIABLE_VERIFICATION.md archive/
mv GYMNASIUM_AND_AIRL_GUIDE.md archive/
mv PHASE2_VALIDATION_CHECKLIST.md archive/
mv DEPTH_INTEGRATION_DETAILED.md archive/
mv OPTION_A_VS_B.md archive/
mv OPTION_DIFFERENCE_SIMPLE.md archive/
mv OPTION_A_DEPTH_H_EXPLAINED.md archive/
mv RESEARCH_DISCUSSION.md archive/
mv AIRL_COMPLETE_GUIDE.md archive/
mv PROJECT_SUMMARY.md archive/
mv FOLDER_STRUCTURE.md archive/

# Keep in root:
# - README.md
# - PROJECT_OVERVIEW.md
# - CLAUDE.md
# - REORGANIZATION_PLAN.md (this file)
```

---

## 📝 Update Required

After moving files, update links in:

### README.md
```markdown
# Before
[AIRL_DESIGN.md](AIRL_DESIGN.md)

# After
[AIRL_DESIGN.md](docs/AIRL_DESIGN.md)
```

### PROJECT_OVERVIEW.md
```markdown
# Before
[AIRL_DESIGN.md](AIRL_DESIGN.md)

# After
[AIRL_DESIGN.md](docs/AIRL_DESIGN.md)
```

### Other documents
- Check all markdown files for internal links
- Update paths as needed

---

## ✅ Benefits

**Before**: 19 files in root → confusing
**After**: 3 files in root → clear entry point

**Structure**:
- ✅ Clear hierarchy (start → core → reference → archive)
- ✅ Easy navigation for new lab members
- ✅ Reduced clutter
- ✅ Historical context preserved

**Documentation Quality**:
- ✅ Current documents easily identifiable
- ✅ Outdated content archived (not deleted)
- ✅ Single source of truth for each topic

---

## 🚀 Next Steps

1. **Review this plan** - Confirm with team
2. **Execute migration** - Run commands above
3. **Update links** - Fix all internal references
4. **Test navigation** - Verify all links work
5. **Update README** - Reflect new structure
6. **Announce changes** - Notify lab members

---

## 📋 Checklist

- [ ] Review reorganization plan
- [ ] Backup current state (git commit)
- [ ] Create folders (docs/, progress/, archive/)
- [ ] Move files to new locations
- [ ] Update all internal links
- [ ] Update README.md references
- [ ] Update PROJECT_OVERVIEW.md references
- [ ] Test all markdown links
- [ ] Commit reorganized structure
- [ ] Notify lab members

---

**Created**: 2025-12-26
**Status**: Ready for implementation
**Approval**: Pending user review
