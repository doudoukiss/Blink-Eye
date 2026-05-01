"""Continuity-memory compatibility surface with lazy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AutobiographyService",
    "BrainAutobiographicalEntryRecord",
    "BrainAutobiographyEntryKind",
    "BrainMultimodalAutobiographyModality",
    "BrainMultimodalAutobiographyPrivacyClass",
    "BrainMultimodalAutobiographyRecord",
    "BrainClaimCurrentnessStatus",
    "BrainClaimEvidenceRecord",
    "BrainClaimGovernanceProjection",
    "BrainClaimGovernanceRecord",
    "BrainClaimRecord",
    "BrainClaimRetentionClass",
    "BrainClaimReviewState",
    "BrainClaimSupersessionRecord",
    "BrainClaimTemporalMode",
    "BrainContinuityQuery",
    "BrainContinuityDossierContradiction",
    "BrainContinuityDossierAvailability",
    "BrainContinuityDossierEvidenceRef",
    "BrainContinuityDossierFactRecord",
    "BrainContinuityDossierFreshness",
    "BrainContinuityDossierGovernanceRecord",
    "BrainContinuityDossierIssueRecord",
    "BrainContinuityDossierKind",
    "BrainContinuityDossierProjection",
    "BrainContinuityDossierRecord",
    "BrainContinuityDossierTaskAvailability",
    "BrainContinuityGraphEdgeKind",
    "BrainContinuityGraphEdgeRecord",
    "BrainContinuityGraphNodeKind",
    "BrainContinuityGraphNodeRecord",
    "BrainContinuityGraphProjection",
    "BrainContinuitySearchResult",
    "BrainCoreMemoryBlockKind",
    "BrainCoreMemoryBlockRecord",
    "BrainEntityRecord",
    "BrainMemoryHealthReportRecord",
    "BrainMemoryPalaceRecord",
    "BrainMemoryPalaceSnapshot",
    "DiscourseEpisode",
    "DiscourseEpisodeMemoryRef",
    "DiscourseEpisodeV3Collector",
    "MemoryCommandIntent",
    "MemoryContinuityDiscourseEpisodeRef",
    "MemoryContinuitySelectedRef",
    "MemoryContinuitySuppressedReason",
    "MemoryContinuityTrace",
    "MemoryContinuityV3",
    "BrainMemoryUseTrace",
    "BrainMemoryUseTraceRef",
    "BrainProceduralExecutionTraceRecord",
    "BrainProceduralActivationConditionRecord",
    "BrainProceduralEffectRecord",
    "BrainProceduralFailureSignatureRecord",
    "BrainProceduralInvariantRecord",
    "BrainProceduralOutcomeKind",
    "BrainProceduralOutcomeRecord",
    "BrainProceduralSkillProjection",
    "BrainProceduralSkillRecord",
    "BrainProceduralSkillStatsRecord",
    "BrainProceduralSkillStatus",
    "BrainProceduralStepTraceRecord",
    "BrainSkillDemotionProposal",
    "BrainSkillEvidenceLedger",
    "BrainSkillEvidenceRecord",
    "BrainSkillGovernanceProjection",
    "BrainSkillGovernanceStatus",
    "BrainSkillPromotionProposal",
    "BrainSkillScenarioCoverage",
    "BrainProceduralTraceProjection",
    "BrainProceduralTraceStatus",
    "BrainReflectionAutobiographyDraft",
    "BrainReflectionAutobiographyUpdate",
    "BrainReflectionBlockUpdate",
    "BrainReflectionClaimCandidate",
    "BrainReflectionClaimDecision",
    "BrainReflectionCycleRecord",
    "BrainReflectionCycleResult",
    "BrainReflectionDraft",
    "BrainReflectionEngine",
    "BrainReflectionEngineConfig",
    "BrainReflectionHealthFinding",
    "BrainReflectionReconciliationDecision",
    "BrainReflectionRunResult",
    "BrainReflectionScheduler",
    "BrainReflectionSchedulerConfig",
    "BrainReflectionSegment",
    "BrainReflectionWorker",
    "BrainReflectionWorkerConfig",
    "BrainGovernanceReasonCode",
    "BrainMemoryGovernanceActionResult",
    "ClaimLedger",
    "ContinuityRetriever",
    "CoreMemoryBlockService",
    "EntityRegistry",
    "MemoryHealthService",
    "build_continuity_dossier_projection",
    "build_continuity_graph_projection",
    "build_claim_governance_projection",
    "build_multimodal_autobiography_digest",
    "build_memory_palace_snapshot",
    "build_memory_continuity_trace",
    "build_memory_use_trace",
    "compile_discourse_episode_v3",
    "compile_discourse_episode_v3_from_actor_events",
    "detect_memory_command_intent",
    "discourse_episode_counts_by_category",
    "display_summary_for_memory_ref",
    "expand_bilingual_memory_query",
    "apply_memory_governance_action",
    "build_skill_evidence_inspection",
    "build_skill_evidence_ledger",
    "build_skill_governance_projection",
    "build_procedural_skill_projection",
    "build_procedural_trace_projection",
    "distill_scene_episode",
    "display_kind_for_claim_predicate",
    "parse_multimodal_autobiography_record",
    "render_claim_summary",
    "render_public_memory_provenance",
    "render_safe_memory_provenance_label",
    "public_actor_event_metadata_for_discourse_episode",
    "public_actor_event_metadata_for_memory_continuity",
    "stamp_memory_continuity_trace",
    "stamp_memory_use_trace",
]

_EXPORTS = {
    "AutobiographyService": (
        "blink.brain.memory_v2.autobiography",
        "AutobiographyService",
    ),
    "BrainAutobiographicalEntryRecord": (
        "blink.brain.memory_v2.autobiography",
        "BrainAutobiographicalEntryRecord",
    ),
    "BrainAutobiographyEntryKind": (
        "blink.brain.memory_v2.autobiography",
        "BrainAutobiographyEntryKind",
    ),
    "BrainMultimodalAutobiographyModality": (
        "blink.brain.memory_v2.multimodal_autobiography",
        "BrainMultimodalAutobiographyModality",
    ),
    "BrainMultimodalAutobiographyPrivacyClass": (
        "blink.brain.memory_v2.multimodal_autobiography",
        "BrainMultimodalAutobiographyPrivacyClass",
    ),
    "BrainMultimodalAutobiographyRecord": (
        "blink.brain.memory_v2.multimodal_autobiography",
        "BrainMultimodalAutobiographyRecord",
    ),
    "BrainClaimEvidenceRecord": ("blink.brain.memory_v2.claims", "BrainClaimEvidenceRecord"),
    "BrainClaimCurrentnessStatus": ("blink.brain.projections", "BrainClaimCurrentnessStatus"),
    "BrainClaimGovernanceProjection": (
        "blink.brain.projections",
        "BrainClaimGovernanceProjection",
    ),
    "BrainClaimGovernanceRecord": ("blink.brain.projections", "BrainClaimGovernanceRecord"),
    "BrainClaimRecord": ("blink.brain.memory_v2.claims", "BrainClaimRecord"),
    "BrainClaimRetentionClass": ("blink.brain.projections", "BrainClaimRetentionClass"),
    "BrainClaimReviewState": ("blink.brain.projections", "BrainClaimReviewState"),
    "BrainClaimSupersessionRecord": (
        "blink.brain.memory_v2.claims",
        "BrainClaimSupersessionRecord",
    ),
    "BrainClaimTemporalMode": ("blink.brain.memory_v2.claims", "BrainClaimTemporalMode"),
    "ClaimLedger": ("blink.brain.memory_v2.claims", "ClaimLedger"),
    "render_claim_summary": ("blink.brain.memory_v2.claims", "render_claim_summary"),
    "BrainGovernanceReasonCode": ("blink.brain.projections", "BrainGovernanceReasonCode"),
    "build_claim_governance_projection": (
        "blink.brain.memory_v2.governance",
        "build_claim_governance_projection",
    ),
    "build_multimodal_autobiography_digest": (
        "blink.brain.memory_v2.multimodal_autobiography",
        "build_multimodal_autobiography_digest",
    ),
    "distill_scene_episode": (
        "blink.brain.memory_v2.multimodal_autobiography",
        "distill_scene_episode",
    ),
    "parse_multimodal_autobiography_record": (
        "blink.brain.memory_v2.multimodal_autobiography",
        "parse_multimodal_autobiography_record",
    ),
    "BrainCoreMemoryBlockKind": (
        "blink.brain.memory_v2.core_blocks",
        "BrainCoreMemoryBlockKind",
    ),
    "BrainCoreMemoryBlockRecord": (
        "blink.brain.memory_v2.core_blocks",
        "BrainCoreMemoryBlockRecord",
    ),
    "CoreMemoryBlockService": ("blink.brain.memory_v2.core_blocks", "CoreMemoryBlockService"),
    "BrainContinuityDossierContradiction": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierContradiction",
    ),
    "BrainContinuityDossierAvailability": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierAvailability",
    ),
    "BrainContinuityDossierEvidenceRef": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierEvidenceRef",
    ),
    "BrainContinuityDossierFactRecord": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierFactRecord",
    ),
    "BrainContinuityDossierFreshness": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierFreshness",
    ),
    "BrainContinuityDossierGovernanceRecord": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierGovernanceRecord",
    ),
    "BrainContinuityDossierIssueRecord": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierIssueRecord",
    ),
    "BrainContinuityDossierKind": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierKind",
    ),
    "BrainContinuityDossierProjection": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierProjection",
    ),
    "BrainContinuityDossierRecord": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierRecord",
    ),
    "BrainContinuityDossierTaskAvailability": (
        "blink.brain.memory_v2.dossiers",
        "BrainContinuityDossierTaskAvailability",
    ),
    "build_continuity_dossier_projection": (
        "blink.brain.memory_v2.dossiers",
        "build_continuity_dossier_projection",
    ),
    "BrainEntityRecord": ("blink.brain.memory_v2.entities", "BrainEntityRecord"),
    "EntityRegistry": ("blink.brain.memory_v2.entities", "EntityRegistry"),
    "BrainContinuityGraphEdgeKind": (
        "blink.brain.memory_v2.graph",
        "BrainContinuityGraphEdgeKind",
    ),
    "BrainContinuityGraphEdgeRecord": (
        "blink.brain.memory_v2.graph",
        "BrainContinuityGraphEdgeRecord",
    ),
    "BrainContinuityGraphNodeKind": (
        "blink.brain.memory_v2.graph",
        "BrainContinuityGraphNodeKind",
    ),
    "BrainContinuityGraphNodeRecord": (
        "blink.brain.memory_v2.graph",
        "BrainContinuityGraphNodeRecord",
    ),
    "BrainContinuityGraphProjection": (
        "blink.brain.memory_v2.graph",
        "BrainContinuityGraphProjection",
    ),
    "build_continuity_graph_projection": (
        "blink.brain.memory_v2.graph",
        "build_continuity_graph_projection",
    ),
    "BrainMemoryHealthReportRecord": (
        "blink.brain.memory_v2.health",
        "BrainMemoryHealthReportRecord",
    ),
    "MemoryHealthService": ("blink.brain.memory_v2.health", "MemoryHealthService"),
    "BrainMemoryPalaceRecord": (
        "blink.brain.memory_v2.memory_palace",
        "BrainMemoryPalaceRecord",
    ),
    "BrainMemoryPalaceSnapshot": (
        "blink.brain.memory_v2.memory_palace",
        "BrainMemoryPalaceSnapshot",
    ),
    "build_memory_palace_snapshot": (
        "blink.brain.memory_v2.memory_palace",
        "build_memory_palace_snapshot",
    ),
    "DiscourseEpisode": ("blink.brain.memory_v2.discourse_episode", "DiscourseEpisode"),
    "DiscourseEpisodeMemoryRef": (
        "blink.brain.memory_v2.discourse_episode",
        "DiscourseEpisodeMemoryRef",
    ),
    "DiscourseEpisodeV3Collector": (
        "blink.brain.memory_v2.discourse_episode",
        "DiscourseEpisodeV3Collector",
    ),
    "compile_discourse_episode_v3": (
        "blink.brain.memory_v2.discourse_episode",
        "compile_discourse_episode_v3",
    ),
    "compile_discourse_episode_v3_from_actor_events": (
        "blink.brain.memory_v2.discourse_episode",
        "compile_discourse_episode_v3_from_actor_events",
    ),
    "discourse_episode_counts_by_category": (
        "blink.brain.memory_v2.discourse_episode",
        "discourse_episode_counts_by_category",
    ),
    "public_actor_event_metadata_for_discourse_episode": (
        "blink.brain.memory_v2.discourse_episode",
        "public_actor_event_metadata_for_discourse_episode",
    ),
    "MemoryCommandIntent": (
        "blink.brain.memory_v2.continuity_trace",
        "MemoryCommandIntent",
    ),
    "MemoryContinuityDiscourseEpisodeRef": (
        "blink.brain.memory_v2.continuity_trace",
        "MemoryContinuityDiscourseEpisodeRef",
    ),
    "MemoryContinuitySelectedRef": (
        "blink.brain.memory_v2.continuity_trace",
        "MemoryContinuitySelectedRef",
    ),
    "MemoryContinuitySuppressedReason": (
        "blink.brain.memory_v2.continuity_trace",
        "MemoryContinuitySuppressedReason",
    ),
    "MemoryContinuityTrace": (
        "blink.brain.memory_v2.continuity_trace",
        "MemoryContinuityTrace",
    ),
    "MemoryContinuityV3": (
        "blink.brain.memory_v2.continuity_trace",
        "MemoryContinuityV3",
    ),
    "build_memory_continuity_trace": (
        "blink.brain.memory_v2.continuity_trace",
        "build_memory_continuity_trace",
    ),
    "detect_memory_command_intent": (
        "blink.brain.memory_v2.continuity_trace",
        "detect_memory_command_intent",
    ),
    "display_summary_for_memory_ref": (
        "blink.brain.memory_v2.continuity_trace",
        "display_summary_for_memory_ref",
    ),
    "expand_bilingual_memory_query": (
        "blink.brain.memory_v2.continuity_trace",
        "expand_bilingual_memory_query",
    ),
    "public_actor_event_metadata_for_memory_continuity": (
        "blink.brain.memory_v2.continuity_trace",
        "public_actor_event_metadata_for_memory_continuity",
    ),
    "stamp_memory_continuity_trace": (
        "blink.brain.memory_v2.continuity_trace",
        "stamp_memory_continuity_trace",
    ),
    "BrainMemoryUseTrace": (
        "blink.brain.memory_v2.use_trace",
        "BrainMemoryUseTrace",
    ),
    "BrainMemoryUseTraceRef": (
        "blink.brain.memory_v2.use_trace",
        "BrainMemoryUseTraceRef",
    ),
    "build_memory_use_trace": (
        "blink.brain.memory_v2.use_trace",
        "build_memory_use_trace",
    ),
    "display_kind_for_claim_predicate": (
        "blink.brain.memory_v2.use_trace",
        "display_kind_for_claim_predicate",
    ),
    "render_safe_memory_provenance_label": (
        "blink.brain.memory_v2.use_trace",
        "render_safe_memory_provenance_label",
    ),
    "render_public_memory_provenance": (
        "blink.brain.memory_v2.memory_palace",
        "render_public_memory_provenance",
    ),
    "stamp_memory_use_trace": (
        "blink.brain.memory_v2.use_trace",
        "stamp_memory_use_trace",
    ),
    "BrainMemoryGovernanceActionResult": (
        "blink.brain.memory_v2.governance_actions",
        "BrainMemoryGovernanceActionResult",
    ),
    "apply_memory_governance_action": (
        "blink.brain.memory_v2.governance_actions",
        "apply_memory_governance_action",
    ),
    "BrainProceduralExecutionTraceRecord": (
        "blink.brain.memory_v2.procedural",
        "BrainProceduralExecutionTraceRecord",
    ),
    "BrainProceduralActivationConditionRecord": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralActivationConditionRecord",
    ),
    "BrainProceduralEffectRecord": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralEffectRecord",
    ),
    "BrainProceduralFailureSignatureRecord": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralFailureSignatureRecord",
    ),
    "BrainProceduralInvariantRecord": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralInvariantRecord",
    ),
    "BrainProceduralOutcomeKind": (
        "blink.brain.memory_v2.procedural",
        "BrainProceduralOutcomeKind",
    ),
    "BrainProceduralOutcomeRecord": (
        "blink.brain.memory_v2.procedural",
        "BrainProceduralOutcomeRecord",
    ),
    "BrainProceduralStepTraceRecord": (
        "blink.brain.memory_v2.procedural",
        "BrainProceduralStepTraceRecord",
    ),
    "BrainProceduralSkillProjection": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralSkillProjection",
    ),
    "BrainProceduralSkillRecord": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralSkillRecord",
    ),
    "BrainProceduralSkillStatsRecord": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralSkillStatsRecord",
    ),
    "BrainProceduralSkillStatus": (
        "blink.brain.memory_v2.skills",
        "BrainProceduralSkillStatus",
    ),
    "BrainSkillDemotionProposal": (
        "blink.brain.memory_v2.skill_promotion",
        "BrainSkillDemotionProposal",
    ),
    "BrainSkillEvidenceLedger": (
        "blink.brain.memory_v2.skill_evidence",
        "BrainSkillEvidenceLedger",
    ),
    "BrainSkillEvidenceRecord": (
        "blink.brain.memory_v2.skill_evidence",
        "BrainSkillEvidenceRecord",
    ),
    "BrainSkillGovernanceProjection": (
        "blink.brain.memory_v2.skill_promotion",
        "BrainSkillGovernanceProjection",
    ),
    "BrainSkillGovernanceStatus": (
        "blink.brain.memory_v2.skill_promotion",
        "BrainSkillGovernanceStatus",
    ),
    "BrainSkillPromotionProposal": (
        "blink.brain.memory_v2.skill_promotion",
        "BrainSkillPromotionProposal",
    ),
    "BrainSkillScenarioCoverage": (
        "blink.brain.memory_v2.skill_evidence",
        "BrainSkillScenarioCoverage",
    ),
    "BrainProceduralTraceProjection": (
        "blink.brain.memory_v2.procedural",
        "BrainProceduralTraceProjection",
    ),
    "BrainProceduralTraceStatus": (
        "blink.brain.memory_v2.procedural",
        "BrainProceduralTraceStatus",
    ),
    "build_skill_evidence_inspection": (
        "blink.brain.memory_v2.skill_evidence",
        "build_skill_evidence_inspection",
    ),
    "build_skill_evidence_ledger": (
        "blink.brain.memory_v2.skill_evidence",
        "build_skill_evidence_ledger",
    ),
    "build_skill_governance_projection": (
        "blink.brain.memory_v2.skill_promotion",
        "build_skill_governance_projection",
    ),
    "build_procedural_skill_projection": (
        "blink.brain.memory_v2.skills",
        "build_procedural_skill_projection",
    ),
    "build_procedural_trace_projection": (
        "blink.brain.memory_v2.procedural",
        "build_procedural_trace_projection",
    ),
    "BrainReflectionAutobiographyDraft": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionAutobiographyDraft",
    ),
    "BrainReflectionAutobiographyUpdate": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionAutobiographyUpdate",
    ),
    "BrainReflectionBlockUpdate": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionBlockUpdate",
    ),
    "BrainReflectionClaimCandidate": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionClaimCandidate",
    ),
    "BrainReflectionClaimDecision": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionClaimDecision",
    ),
    "BrainReflectionCycleRecord": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionCycleRecord",
    ),
    "BrainReflectionCycleResult": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionCycleResult",
    ),
    "BrainReflectionDraft": ("blink.brain.memory_v2.reflection", "BrainReflectionDraft"),
    "BrainReflectionEngine": ("blink.brain.memory_v2.reflection", "BrainReflectionEngine"),
    "BrainReflectionEngineConfig": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionEngineConfig",
    ),
    "BrainReflectionHealthFinding": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionHealthFinding",
    ),
    "BrainReflectionReconciliationDecision": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionReconciliationDecision",
    ),
    "BrainReflectionRunResult": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionRunResult",
    ),
    "BrainReflectionScheduler": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionScheduler",
    ),
    "BrainReflectionSchedulerConfig": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionSchedulerConfig",
    ),
    "BrainReflectionSegment": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionSegment",
    ),
    "BrainReflectionWorker": ("blink.brain.memory_v2.reflection", "BrainReflectionWorker"),
    "BrainReflectionWorkerConfig": (
        "blink.brain.memory_v2.reflection",
        "BrainReflectionWorkerConfig",
    ),
    "BrainContinuityQuery": ("blink.brain.memory_v2.retrieval", "BrainContinuityQuery"),
    "BrainContinuitySearchResult": (
        "blink.brain.memory_v2.retrieval",
        "BrainContinuitySearchResult",
    ),
    "ContinuityRetriever": ("blink.brain.memory_v2.retrieval", "ContinuityRetriever"),
}


def __getattr__(name: str):
    """Resolve compatibility exports lazily to avoid import-time cycles."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose stable lazy-export names to interactive tooling."""
    return sorted(set(globals()) | set(__all__))
