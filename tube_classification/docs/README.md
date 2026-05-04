# Documentation Structure

This directory organizes all project documentation into logical categories for ease of navigation and maintenance.

## Structure

```
docs/
├── README.md                          # This file
├── PREPROCESSING_CONFIG_GUIDE.md       # Configuration reference
├── guides/                             # Setup and how-to guides
│   ├── MOBILESAM_SETUP.md             # MobileSAM model setup
│   ├── IMPLEMENTATION_GUIDE.md        # Step-by-step implementation
│   ├── IMAGE_QUALITY_OPTIMIZATION.md  # Quality tuning and analysis
│   ├── HD_CAMERA_OPTIMIZATION.md      # Camera calibration and setup
│   ├── GUIDED_FILTER_COMPLETE.md      # Guided filter reference
│   ├── GUIDED_FILTER_INTEGRATION_FINAL.md
│   └── GUIDED_FILTER_INTEGRATION_SUMMARY.py
├── troubleshooting/                   # Known issues and solutions
│   ├── CAMERA_TIMEOUT_FIX.md          # Camera/USB issues
│   ├── PERFORMANCE_FIX.md             # Performance tuning
│   ├── HD_CAPTURE_GUARANTEE.md        # HD capture quality assurance
│   └── HD_CAPTURE_VERIFICATION.md     # Verification procedures
└── archive/                           # Historical/milestone docs
    ├── CHECKLIST.md                   # Project checklist
    ├── DELIVERABLES_SUMMARY.md        # Final deliverables
    ├── PROJECT_COMPLETE.md            # Completion summary
    ├── IMPLEMENTATION_READY.md        # Readiness gate
    ├── INDEX_AND_NAVIGATION.md        # Old navigation guide
    └── HD_UPGRADE_COMPLETE.md         # Release summary
```

## Quick Links

**Getting Started:**
- [README.md](../README.md) - Main project overview
- [QUICK_START.md](../QUICK_START.md) - Quick reference

**Setup & Implementation:**
- [guides/MOBILESAM_SETUP.md](guides/MOBILESAM_SETUP.md)
- [guides/IMPLEMENTATION_GUIDE.md](guides/IMPLEMENTATION_GUIDE.md)

**Configuration & Tuning:**
- [PREPROCESSING_CONFIG_GUIDE.md](PREPROCESSING_CONFIG_GUIDE.md)
- [guides/IMAGE_QUALITY_OPTIMIZATION.md](guides/IMAGE_QUALITY_OPTIMIZATION.md)
- [guides/HD_CAMERA_OPTIMIZATION.md](guides/HD_CAMERA_OPTIMIZATION.md)

**Troubleshooting:**
- [troubleshooting/CAMERA_TIMEOUT_FIX.md](troubleshooting/CAMERA_TIMEOUT_FIX.md)
- [troubleshooting/PERFORMANCE_FIX.md](troubleshooting/PERFORMANCE_FIX.md)

## Documentation Maintenance

### Updating Docs
1. Check which category your content belongs to
2. Place files in appropriate subdirectory
3. Update this README if adding new sections
4. Avoid duplicate content; link instead

### Archiving Docs
When a document becomes outdated:
1. Move to `archive/` directory
2. Add date and reason in filename or README
3. Update links in other docs to point to archive if still needed

## Historical Note

This structure was implemented as part of industrial-grade project reorganization to improve maintainability and clarity.
