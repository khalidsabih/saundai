"""Central version and schema identifiers for DJAI."""

DJAI_VERSION = "2.0.0-rc.1"
MUSICAL_TIMELINE_SCHEMA_VERSION = "1.1.1"
# Neural perception evidence can change rhythm/representation state, so 5.0
# intentionally invalidates older analysis caches. Performance layers remain
# independently cached by source fingerprint.
CACHE_SCHEMA_VERSION = "5.0.0"
PLAN_SCHEMA_VERSION = "8.0.0"
DECISION_VERSION = "8.0.0"
PLANNER_VERSION = "8.0.0"
SET_PLAN_VERSION = "8.0.0"
MEASUREMENT_VERSION = "3.0.0"
FEEDBACK_SCHEMA_VERSION = "2.0"
PERSONALIZATION_VERSION = "2.0"
