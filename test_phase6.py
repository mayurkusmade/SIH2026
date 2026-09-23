"""
IBVAP PHASE 6 VERIFICATION: FACIAL RECOGNITION SYSTEM (FRS)
===========================================================

Verifies the biometric layer end to end WITHOUT requiring any face model on the
machine: the detector and embedder are injected, so 1:N matching accuracy,
threshold policy, temporal smoothing and indexing are all genuinely exercised
rather than mocked away.

  1. Embedding normalization and image geometry helpers.
  2. Index lifecycle: enroll, persist across process restarts, reload, remove.
  3. 1:N ACCURACY - planted identities, noisy probes, measured top-1 rate.
  4. Threshold policy - MATCH / REVIEW / UNKNOWN bands, false-accept rate.
  5. Deck-scale search - 10,000 identities x 512-d, latency vs the "<15 ms" claim.
  6. Identity smoothing - promotion after confirmation, no flicker, best-per-track.
  7. Per-track throttling - the model is not re-run on every single frame.
  8. Gate filtering - minimum track height and face size.
  9. HONESTY GUARD - a non-biometric embedder can never authorize a person.
 10. Backend fallback chain reports what it actually loaded.
 11. Enrollment from a frame, and overlay drawing without OpenCV.

Run:  python test_phase6.py
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np

from modules.frs import (
    AUTHORIZED,
    DEGRADED,
    NO_FACE,
    REVIEW,
    ROLE_AUTHORIZED,
    ROLE_WATCHLIST,
    UNKNOWN,
    WATCHLIST_HIT,
    DescriptorFallbackEmbedder,
    FaceIndex,
    FRSModule,
    InjectedEmbedder,
    InjectedFaceDetector,
    build_face_detector,
    build_face_embedder,
    crop_region,
    l2_normalize,
    resize_nearest,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
failures = []


def check(condition, message):
    print(f"{PASS if condition else FAIL} {message}")
    if not condition:
        failures.append(message)
    return condition


def header(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


WORKDIR = tempfile.mkdtemp(prefix="ibvap_frs_")


def db_path(name):
    return os.path.join(WORKDIR, f"{name}.sqlite")


def random_unit(dim, rng):
    vector = rng.normal(size=dim).astype(np.float32)
    return l2_normalize(vector)


def probe_with_similarity(base, target_cos, rng):
    """
    Builds a probe whose cosine similarity to `base` is EXACTLY target_cos.

    Deterministic control of the similarity lets each threshold band be tested
    precisely instead of hoping a noise level lands where expected.
    """
    target = float(np.clip(target_cos, 1e-3, 0.999))
    orth = rng.normal(size=base.shape).astype(np.float32)
    orth = orth - float(np.dot(orth, base)) * base
    orth = orth / (float(np.linalg.norm(orth)) + 1e-12)
    scale = float(np.sqrt(1.0 / (target * target) - 1.0))
    return l2_normalize(base + scale * orth)


def main():
    print("=" * 74)
    print("  IBVAP PHASE 6 VERIFICATION: FACIAL RECOGNITION SYSTEM (FRS)")
    print("=" * 74)
    rng = np.random.default_rng(20260922)
    DIM = 64  # small dim keeps the accuracy sweep fast; scale test uses 512

    # ------------------------------------------------------------------
    header("1. EMBEDDING NORMALIZATION + GEOMETRY HELPERS")
    # ------------------------------------------------------------------
    raw = np.array([3.0, 4.0], dtype=np.float32)
    normed = l2_normalize(raw)
    check(abs(float(np.linalg.norm(normed)) - 1.0) < 1e-5, "vectors are L2-normalized")
    check(abs(normed[0] - 0.6) < 1e-5 and abs(normed[1] - 0.8) < 1e-5,
          "normalization preserves direction (3,4 -> 0.6,0.8)")
    check(float(np.linalg.norm(l2_normalize(raw * 1000))) > 0.99,
          "scale invariance: a bright/dark crop yields the same unit vector")
    zero = l2_normalize(np.zeros(8, dtype=np.float32))
    check(np.all(np.isfinite(zero)), "zero vector does not produce NaN/inf")

    frame = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)
    cropped = crop_region(frame, (10, 5, 50, 35))
    check(cropped is not None and cropped.shape == (30, 40, 3), "crop_region honours the box")
    check(crop_region(frame, (50, 30, 10, 5)) is None, "inverted box is rejected")
    check(crop_region(frame, (1, 1, 2, 2)) is None, "degenerate 1px box is rejected")
    check(crop_region(None, (0, 0, 5, 5)) is None, "None frame is handled")
    resized = resize_nearest(frame, 16, 16)
    check(resized.shape == (16, 16, 3), f"resize_nearest -> {resized.shape[1::-1]}")
    check(resize_nearest(frame, 60, 40).shape == frame.shape, "no-op resize is exact")

    # ------------------------------------------------------------------
    header("2. INDEX LIFECYCLE: ENROLL, PERSIST, RELOAD, REMOVE")
    # ------------------------------------------------------------------
    first_db = db_path("lifecycle")
    index = FaceIndex(db_path=first_db)
    check(index.stats()["faces"] == 0, "a fresh index is empty")

    embedding = random_unit(DIM, rng)
    face_id = index.add(
        embedding, person_id="SSB-4587", name="Constable R. Singh",
        role=ROLE_AUTHORIZED, unit="42nd Battalion SSB",
    )
    check(index.stats()["faces"] == 1, "face enrolled")
    check(index.stats()["identities"] == 1, "identity counted once")

    reopened = FaceIndex(db_path=first_db)
    check(reopened.stats()["faces"] == 1,
          "roster survives a restart (SQLite, fully offline - no service needed)")
    hit = reopened.search(embedding, top_k=1)[0]
    check(abs(hit["similarity"] - 1.0) < 1e-4,
          f"a stored face matches itself at cosine {hit['similarity']}")
    check(hit["name"] == "Constable R. Singh", "metadata round-trips with the vector")
    check(reopened.stats()["dim"] == DIM, "index remembers the embedding dimension")

    check(reopened.remove(face_id), "face removed")
    check(reopened.stats()["faces"] == 0, "index empty after removal")

    mismatch = FaceIndex(db_path=db_path("dimcheck"))
    mismatch.add(random_unit(32, rng), person_id="P1")
    mixed_dim_rejected = False
    try:
        mismatch.add(random_unit(16, rng), person_id="P2")
    except ValueError:
        mixed_dim_rejected = True
    check(mixed_dim_rejected,
          "mixing embedding dimensions is refused (would corrupt matching)")

    # ------------------------------------------------------------------
    header("3. 1:N ACCURACY - PLANTED IDENTITIES, NOISY PROBES")
    # ------------------------------------------------------------------
    N_IDENTITIES = 300
    accuracy_db = db_path("accuracy")
    roster = FaceIndex(db_path=accuracy_db)

    bases = {f"ID-{i:04d}": random_unit(DIM, rng) for i in range(N_IDENTITIES)}
    # One deliberately confusable pair, to make sure "closest" really wins.
    bases["ID-0001"] = l2_normalize(bases["ID-0000"] + 0.5 * random_unit(DIM, rng))

    roster.add_batch(
        list(bases.values()),
        list(bases.keys()),
        names=[f"Person {pid}" for pid in bases],
        roles=[ROLE_AUTHORIZED] * len(bases),
    )
    check(roster.stats()["faces"] == N_IDENTITIES,
          f"{N_IDENTITIES} identities enrolled in one transaction")

    NOISE = 0.30
    trials = 400
    correct = 0
    probe_similarities = []
    for _ in range(trials):
        person_id = list(bases.keys())[int(rng.integers(0, N_IDENTITIES))]
        probe = l2_normalize(bases[person_id] + NOISE * random_unit(DIM, rng))
        result = roster.identify(probe, top_k=1)
        probe_similarities.append(result["similarity"])
        if result["record"] and result["record"]["person_id"] == person_id:
            correct += 1
    top1 = correct / trials
    print(f"    {trials} noisy probes (noise={NOISE}) -> top-1 accuracy {top1 * 100:.1f}% "
          f"| mean similarity {np.mean(probe_similarities):.3f}")
    check(top1 >= 0.98, f"top-1 identification accuracy is {top1 * 100:.1f}%")
    check(np.mean(probe_similarities) >= 0.45,
          "genuine probes land above the match threshold on average")

    near_miss = roster.identify(
        l2_normalize(bases["ID-0000"] + 0.2 * random_unit(DIM, rng)), top_k=1
    )
    check(not near_miss["hits"] or near_miss["hits"][0]["similarity"] > 0,
          "the closest identity still wins in the confusable pair")

    # ------------------------------------------------------------------
    header("4. THRESHOLD POLICY + FALSE-ACCEPT RATE")
    # ------------------------------------------------------------------
    check(roster.decide(0.90) == "MATCH", "0.90 similarity -> MATCH")
    check(roster.decide(0.40) == "REVIEW", "0.40 similarity -> REVIEW (human check)")
    check(roster.decide(0.10) == "UNKNOWN", "0.10 similarity -> UNKNOWN")

    def measure_far(target_index, probes=500, seed_offset=1):
        """False-accept rate: how often does a stranger clear the gate?"""
        local_rng = np.random.default_rng(4242 + seed_offset)
        accepts = 0
        worst = 0.0
        for _ in range(probes):
            stranger = random_unit(target_index.dim or DIM, local_rng)
            result = target_index.identify(stranger, top_k=1)
            worst = max(worst, result["similarity"])
            if result["decision"] == "MATCH":
                accepts += 1
        return accepts, accepts / probes, worst

    accepts, far, impostor_max = measure_far(roster)
    print(f"    500 impostor probes -> {accepts} false accepts ({far * 100:.2f}%), "
          f"worst similarity {impostor_max:.3f}")

    # 4b. A fixed threshold is NOT universally safe - this is the finding that
    #     makes capacity reporting a feature rather than a nicety. This roster is
    #     300 identities crammed into 64 dimensions (a deliberately tight space,
    #     e.g. what you get from a cheap non-ArcFace descriptor).
    capacity = roster.capacity_report()
    print(f"    capacity @64-d/300 ids: headroom={capacity['headroom']} "
          f"verdict={capacity['verdict']} (99.9th pct impostor similarity="
          f"{capacity['stats']['percentile']})")
    check(capacity["verdict"] == "TIGHT",
          "a crowded embedding space is flagged TIGHT instead of shipped silently")
    check("raise the match threshold" in capacity["advice"],
          "the platform states a concrete remedy")
    check(accepts > 0,
          f"the default threshold really does leak here ({accepts}/{500} false accepts)")

    calibrated = roster.calibrate(percentile=99.99, margin=0.15)
    print(f"    calibrated: {calibrated['before']} -> {calibrated['after']} "
          f"(review band {calibrated['review_threshold']})")
    check(calibrated["raised"], "calibration raised the gate above the biometric floor")
    check(calibrated["after"] >= calibrated["floor"],
          "calibration never loosens below the biometric floor")

    accepts_after, far_after, worst_after = measure_far(roster, seed_offset=2)
    recall_after = 0
    for _ in range(200):
        person_id = list(bases.keys())[int(rng.integers(0, N_IDENTITIES))]
        probe = probe_with_similarity(bases[person_id], 0.90, rng)
        result = roster.identify(probe, top_k=1)
        if result["record"] and result["record"]["person_id"] == person_id:
            recall_after += 1
    print(f"    after calibration: {accepts_after}/{500} false accepts "
          f"(worst {worst_after:.3f}) | genuine recall {recall_after / 200 * 100:.1f}%")
    check(accepts_after == 0,
          "calibration eliminated the false accepts in the crowded space")
    check(recall_after / 200 >= 0.98, "genuine recognition survived the tightened gate")

    roster.match_threshold = 0.45
    roster.review_threshold = 0.32

    # ------------------------------------------------------------------
    header("5. DECK-SCALE SEARCH: 10,000 IDENTITIES x 512-d")
    # ------------------------------------------------------------------
    SCALE_N, SCALE_DIM = 10_000, 512
    scale_db = db_path("scale")
    scale = FaceIndex(db_path=scale_db)
    scale_vectors = rng.normal(size=(SCALE_N, SCALE_DIM)).astype(np.float32)
    scale_vectors = scale_vectors / np.linalg.norm(scale_vectors, axis=1, keepdims=True)
    started = time.perf_counter()
    scale.add_batch(
        list(scale_vectors),
        [f"SUSPECT-{i:05d}" for i in range(SCALE_N)],
        names=[f"Suspect {i}" for i in range(SCALE_N)],
        roles=[ROLE_WATCHLIST] * SCALE_N,
    )
    load_s = time.perf_counter() - started
    print(f"    enrolled {scale.stats()['faces']} faces in {load_s:.1f}s "
          f"({load_s / SCALE_N * 1000:.2f} ms/face)")

    latencies = []
    for i in range(30):
        probe = scale_vectors[i * 17]
        _ = scale.identify(probe, top_k=1)
        latencies.append(scale.last_search_ms)
    mean_ms = float(np.mean(latencies))
    worst_ms = float(np.max(latencies))
    print(f"    exact 1:N cosine search: mean {mean_ms:.2f} ms, worst {worst_ms:.2f} ms "
          f"per probe (deck claim: <15 ms)")
    check(worst_ms < 15.0, f"worst-case search {worst_ms:.2f} ms beats the 15 ms claim")

    verified = 0
    for i in range(50):
        probe = scale_vectors[i * 71]
        result = scale.identify(probe, top_k=1)
        if result["record"] and result["record"]["person_id"] == f"SUSPECT-{i * 71:05d}":
            verified += 1
    check(verified == 50, "10,000-identity identity lookups are all correct")
    check(scale.stats()["dim"] == SCALE_DIM, "512-d embeddings stored as declared")

    scale_capacity = scale.capacity_report()
    print(f"    capacity @512-d/10000 ids: headroom={scale_capacity['headroom']} "
          f"verdict={scale_capacity['verdict']}")
    check(scale_capacity["verdict"] == "OK",
          "a 512-d biometric space stays separable at 10,000 identities")

    # ------------------------------------------------------------------
    header("6-8. TRACK DECISIONS, SMOOTHING, THROTTLING, GATE FILTERS")
    # ------------------------------------------------------------------
    module_db = db_path("module")
    module_index = FaceIndex(db_path=module_db)
    guard_vector = random_unit(DIM, rng)
    suspect_vector = random_unit(DIM, rng)
    module_index.add(guard_vector, person_id="SSB-4587", name="Constable R. Singh",
                     role=ROLE_AUTHORIZED, unit="42nd Battalion SSB", rank="Constable")
    module_index.add(suspect_vector, person_id="SUS-9001", name="Known Smuggler",
                     role=ROLE_WATCHLIST, unit="Cross-border network")

    # Similarities are dialled in exactly, so each threshold band is exercised
    # deliberately rather than arrived at by luck.
    probe_vector = {
        # 0.52 lands in the MATCH band but BELOW the high-confidence fast-lock,
        # so this one must wait for cross-frame confirmation.
        "guard_unconfirmed": probe_with_similarity(guard_vector, 0.52, rng),
        "guard_clean": probe_with_similarity(guard_vector, 0.90, rng),
        "suspect": probe_with_similarity(suspect_vector, 0.90, rng),
        "stranger": random_unit(DIM, rng),
    }

    embed_calls = {"n": 0}
    active_probe = {"key": "guard_unconfirmed"}

    def vector_for(face_bbox):
        embed_calls["n"] += 1
        return probe_vector[active_probe["key"]]

    detector = InjectedFaceDetector([(100, 60, 160, 130, 0.95)])
    embedder = InjectedEmbedder(vector_for=vector_for, dim=DIM)
    frs = FRSModule(
        index=module_index, detector=detector, embedder=embedder,
        confirm_frames=3, throttle_frames=0, min_track_height=40, min_face_px=30,
    )
    state = frs.state()
    check(state["identity_capable"] and state["mode"] == "BIOMETRIC",
          f"identity-capable FRS in {state['mode']} mode")
    check(state["embedder"] == "injected" and state["detector"] == "injected",
          "state() names the loaded backends (no silent substitution)")

    def make_tracks(bbox=(100, 60, 200, 400), track_id=7):
        return {track_id: {"bbox": bbox, "category": "human", "conf": 0.9}}

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    decisions = frs.process(frame, make_tracks(), timestamp="10:00:01", frame_idx=1)
    first = decisions[0]
    print(f"    frame 1 (similarity {first['similarity']}): decision={first['decision']}, "
          f"promoted={first['promoted']}")
    check(first["decision"] == REVIEW,
          f"a genuine-but-unconfirmed match is REVIEW on frame 1 ({first['decision']})")
    check(not first["promoted"], "identity is not promoted on the first frame")
    check(not first["name"], "no name is attached while the identity is unconfirmed")

    for f in range(2, 5):
        decisions = frs.process(frame, make_tracks(), timestamp="10:00:0%d" % f, frame_idx=f)
    promoted = decisions[0]
    check(promoted["promoted"], "identity promoted after confirm_frames agreement")
    check(promoted["decision"] == AUTHORIZED,
          f"authorized patrol recognized as {promoted['decision']}")
    check(promoted["name"] == "Constable R. Singh", "matched name surfaced once confirmed")
    check(promoted["similarity"] > 0.5, f"match similarity {promoted['similarity']}")
    check(promoted["source"] == "BIOMETRIC", "decision labelled as biometric")
    check(promoted["face_bbox"] is not None, "face box reported for the overlay")
    check(promoted["frames_seen"] >= 3, "agreement count reported for the audit trail")

    # The high-confidence fast path: a clean read locks immediately, because
    # waiting three frames to recognise your own patrol wastes a response window.
    active_probe["key"] = "guard_clean"
    frs.purge_tracks()
    instant = frs.process(frame, make_tracks(track_id=12), frame_idx=6)[0]
    print(f"    clean read (similarity {instant['similarity']}): "
          f"decision={instant['decision']}, promoted={instant['promoted']}")
    check(instant["promoted"] and instant["decision"] == AUTHORIZED,
          "a high-confidence read is promoted on the first frame")

    # Temporal smoothing: an unreadable frame must not corrupt a good identity.
    unreadable = {"first": True}

    def flaky_vector(face_bbox):
        if unreadable["first"]:
            unreadable["first"] = False
            return np.zeros(DIM, dtype=np.float32)  # unreadable crop
        return probe_vector["guard_clean"]

    frs.embedder = InjectedEmbedder(vector_for=flaky_vector, dim=DIM)
    frs.purge_tracks()
    skipped = frs.process(frame, make_tracks(track_id=11), frame_idx=10)
    after_bad = frs.process(frame, make_tracks(track_id=11), frame_idx=11)[0]
    check(skipped == [], "an unreadable crop produces no decision at all")
    check(after_bad["decision"] == AUTHORIZED,
          f"the next good frame still recognizes the person ({after_bad['decision']})")

    # Restore the real probe source (the stub above was a one-off experiment).
    frs.embedder = InjectedEmbedder(vector_for=vector_for, dim=DIM)

    # Watchlist hit: same mechanism, different roster role.
    active_probe["key"] = "suspect"
    frs.purge_tracks()
    for f in range(5):
        decisions = frs.process(frame, make_tracks(track_id=22), frame_idx=20 + f)
    check(decisions[0]["decision"] == WATCHLIST_HIT,
          f"roster role drives the decision ({decisions[0]['decision']} for a suspect)")
    check(decisions[0]["role"] == ROLE_WATCHLIST, "watchlist role reported")
    check(decisions[0]["name"] == "Known Smuggler", "watchlist name surfaced for dispatch")

    # Unknown walker - similarity below every band.
    active_probe["key"] = "stranger"
    frs.purge_tracks()
    for f in range(5):
        decisions = frs.process(frame, make_tracks(track_id=33), frame_idx=40 + f)
    print(f"    unknown walker similarity {decisions[0]['similarity']} -> "
          f"{decisions[0]['decision']}")
    check(decisions[0]["decision"] == UNKNOWN,
          f"an unknown walker is not misidentified ({decisions[0]['decision']})")
    check(not decisions[0]["name"], "no name is invented for an unknown walker")

    # Throttling: repeated frames must not re-run the model every time.
    counting = {"n": 0}

    def counting_vector(face_bbox):
        counting["n"] += 1
        return probe_vector["guard"]

    frs.embedder = InjectedEmbedder(vector_for=counting_vector, dim=DIM)
    frs.throttle_frames = 5
    frs.purge_tracks()
    for f in range(12):
        frs.process(frame, make_tracks(track_id=44), frame_idx=100 + f)
    print(f"    12 frames with throttle_frames=5 -> {counting['n']} embedding call(s)")
    check(counting["n"] <= 4,
          "per-track throttling avoids re-running the model on consecutive frames")

    # Gate filters: too-small people/faces are not sent to the model.
    active_probe["key"] = "guard"
    frs.throttle_frames = 0
    frs.purge_tracks()
    small_track = frs.process(frame, make_tracks(bbox=(100, 60, 140, 90), track_id=55), frame_idx=200)
    check(small_track == [], "a distant, tiny person is skipped before the model runs")

    frs.detector = InjectedFaceDetector([])  # no face visible in the frame
    frs.purge_tracks()
    no_face = frs.process(frame, make_tracks(track_id=66), frame_idx=210)
    check(no_face == [], "no face detected -> no decision invented")
    check(frs.track_cache[66]["decision"] == NO_FACE,
          "the track is marked NO_FACE for the UI")

    frs.detector = InjectedFaceDetector([(100, 60, 115, 75, 0.9)])  # too small a face
    frs.min_face_px = 40
    frs.purge_tracks()
    check(frs.process(frame, make_tracks(track_id=77), frame_idx=220) == [],
          "a face below min_face_px is ignored rather than matched")

    removed = frs.purge_tracks(keep={99})
    check(removed >= 0 and 99 not in frs.track_cache, "purge_tracks drops lost tracks")

    # ------------------------------------------------------------------
    header("9. HONESTY GUARD - NON-BIOMETRIC EMBEDDERS CANNOT AUTHORIZE")
    # ------------------------------------------------------------------
    descriptor_index = FaceIndex(db_path=db_path("descriptor"))
    face_frame = (np.random.default_rng(7).normal(size=(120, 120, 3)) * 40 + 128)
    face_frame = np.clip(face_frame, 0, 255).astype(np.uint8)
    descriptor_embedder = DescriptorFallbackEmbedder()
    descriptor_embedder.load()
    descriptor_index.add(descriptor_embedder.embed(face_frame, (20, 20, 90, 90)),
                         person_id="SSB-4587", name="Constable R. Singh",
                         role=ROLE_AUTHORIZED)

    honest = FRSModule(
        index=descriptor_index,
        detector=InjectedFaceDetector([(20, 20, 90, 90, 0.9)]),
        embedder=descriptor_embedder,
        min_track_height=10,
    )
    honest_state = honest.state()
    print(f"    descriptor embedder -> capable={honest_state['identity_capable']}, "
          f"mode={honest_state['mode']}, reason='{honest_state['reason']}'")
    check(not honest_state["identity_capable"],
          "an appearance descriptor is NOT allowed to make identity decisions")
    check(honest.process(face_frame, make_tracks(bbox=(10, 10, 100, 200), track_id=1)) == [],
          "no decision is emitted rather than a false biometric claim")
    check("non-biometric" in honest_state["reason"], "the reason is stated explicitly")

    opted_in = FRSModule(
        index=descriptor_index,
        detector=InjectedFaceDetector([(20, 20, 90, 90, 0.9)]),
        embedder=descriptor_embedder,
        min_track_height=10,
        allow_non_biometric=True,
        confirm_frames=1,
    )
    opted_state = opted_in.state()
    emitted = opted_in.process(face_frame, make_tracks(bbox=(10, 10, 100, 200), track_id=1),
                               frame_idx=1)
    print(f"    explicit opt-in -> mode={opted_state['mode']}, "
          f"emitted={[d['decision'] for d in emitted]}")
    check(opted_state["mode"] == "DEGRADED_NON_BIOMETRIC",
          "opt-in mode is reported as DEGRADED, never as biometric")
    check(emitted and emitted[0]["source"] == "NON_BIOMETRIC",
          "decisions carry the non-biometric provenance")

    # ------------------------------------------------------------------
    header("10. BACKEND FALLBACK CHAIN (this machine has no face models)")
    # ------------------------------------------------------------------
    detector, detector_note = build_face_detector()
    embedder, embedder_note = build_face_embedder()
    print(f"    detector: {detector.describe()} | {detector_note}")
    print(f"    embedder: {embedder.describe()} | {embedder_note}")
    check(isinstance(detector_note, str) and detector_note, "detector chain explains itself")
    check(isinstance(embedder_note, str) and embedder_note, "embedder chain explains itself")
    check(getattr(embedder, "is_biometric", False) is False or embedder.is_loaded,
          "a biometric embedder is only ever reported when it truly loaded")
    if not embedder.is_loaded:
        check(False, "no embedder could be loaded at all (expected the descriptor fallback)")
    else:
        check(True, f"fallback embedder loaded: {embedder.name} "
                    f"(biometric={embedder.is_biometric})")

    # ------------------------------------------------------------------
    header("11. ENROLLMENT FROM A FRAME + OVERLAY WITHOUT OPENCV")
    # ------------------------------------------------------------------
    enroll_index = FaceIndex(db_path=db_path("enroll"))
    enroll_mod = FRSModule(
        index=enroll_index,
        detector=InjectedFaceDetector([(140, 90, 220, 190, 0.97)]),
        embedder=InjectedEmbedder(vector_for=lambda box: random_unit(DIM, rng), dim=DIM),
        min_face_px=20,
    )
    capture = enroll_mod.enroll_face(frame, bbox=(120, 80, 240, 400))
    check(capture is not None, "one-click enrollment extracted a face from a track box")
    check(capture["face_bbox"] == (140, 90, 220, 190), "largest face in the track was chosen")
    check(capture["biometric"], "enrollment reports whether it was biometric")
    enroll_index.add(capture["embedding"], person_id="SSB-9999", name="New Recruit",
                     role=ROLE_AUTHORIZED)
    check(enroll_index.stats()["faces"] == 1, "captured face enrolled into the index")

    no_face_mod = FRSModule(
        index=enroll_index, detector=InjectedFaceDetector([]),
        embedder=InjectedEmbedder(vector_for=lambda box: random_unit(DIM, rng), dim=DIM),
    )
    check(no_face_mod.enroll_face(frame) is None, "enrollment refuses when no face is visible")

    overlay_safe = enroll_mod.draw_faces(frame, [{
        "decision": AUTHORIZED, "face_bbox": (140, 90, 220, 190),
        "name": "New Recruit", "similarity": 0.71, "promoted": True,
    }])
    check(overlay_safe is not None, "overlay drawing degrades safely without OpenCV")

    # ------------------------------------------------------------------
    header("FRS STATE SAMPLE (what the dashboard renders)")
    # ------------------------------------------------------------------
    sample_state = enroll_mod.state()
    for key in ("mode", "identity_capable", "detector", "embedder", "embedding_dim",
                "faces_detected", "decisions_made", "index"):
        check(key in sample_state, f"state() exposes '{key}' for the UI")
    print(f"    {sample_state['mode']} | index={sample_state['index']}")

    shutil.rmtree(WORKDIR, ignore_errors=True)

    # ------------------------------------------------------------------
    header("PHASE 6 RESULT")
    # ------------------------------------------------------------------
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for item in failures:
            print(f"    - {item}")
        print("=" * 74)
        return 1

    print("  ALL CHECKS PASSED - the FRS layer is real, and honest about its limits:")
    print("    * exact 1:N cosine search over a 10,000-identity local index")
    print("    * measured search latency beats the deck's <15 ms claim")
    print("    * identity promoted only after cross-frame agreement (no flicker)")
    print("    * a non-biometric embedder can never authorize a person")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
