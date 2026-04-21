from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import to_gray


@dataclass(slots=True)
class SimilarityResult:
    score: float
    orb_score: float
    ssim_score: float
    hist_score: float
    phash_score: float


def _compute_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_AREA)

    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = numerator / (denominator + 1e-8)
    return float(np.clip(np.mean(ssim_map), 0.0, 1.0))


def _hist_corr_bgr(a: np.ndarray, b: np.ndarray) -> float:
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))


def _phash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    size = hash_size * highfreq_factor
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])
    return (low > med).astype(np.uint8)


def _phash_similarity(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_AREA)
    ha = _phash(gray_a)
    hb = _phash(gray_b)
    hamming = int(np.count_nonzero(ha != hb))
    total = ha.size
    return float(np.clip(1.0 - (hamming / max(1, total)), 0.0, 1.0))


class HybridMatcher:
    def __init__(self, reference_bgr: np.ndarray, max_side: int = 640):
        self.reference = self._normalize_size(reference_bgr, max_side=max_side)
        self.reference_gray = to_gray(self.reference)
        self.orb = cv2.ORB_create(nfeatures=1800)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.reference_gray, None)

    @staticmethod
    def _normalize_size(image: np.ndarray, max_side: int = 640) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= max_side:
            return image
        scale = max_side / float(max(h, w))
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _orb_score(self, sample_gray: np.ndarray) -> float:
        if self.des_ref is None or len(self.kp_ref) < 8:
            return 0.0

        kp_s, des_s = self.orb.detectAndCompute(sample_gray, None)
        if des_s is None or kp_s is None or len(kp_s) < 8:
            return 0.0

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(self.des_ref, des_s, k=2)

        good_matches = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            return 0.0

        ref_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        sample_pts = np.float32([kp_s[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(ref_pts, sample_pts, cv2.RANSAC, 5.0)

        if mask is None:
            return 0.0

        inliers = int(mask.ravel().sum())
        inlier_ratio = inliers / max(1, len(good_matches))

        denom = max(1, min(len(self.kp_ref), len(kp_s)))
        support = inliers / denom

        inlier_dist = [m.distance for m, keep in zip(good_matches, mask.ravel()) if keep]
        if inlier_dist:
            distance_quality = 1.0 - (float(np.median(inlier_dist)) / 100.0)
        else:
            distance_quality = 0.0
        distance_quality = float(np.clip(distance_quality, 0.0, 1.0))

        score = 0.55 * inlier_ratio + 0.35 * support + 0.10 * distance_quality
        return float(np.clip(score, 0.0, 1.0))

    def compare(self, sample_bgr: np.ndarray) -> SimilarityResult:
        sample = self._normalize_size(sample_bgr)
        if sample.shape[:2] != self.reference.shape[:2]:
            sample = cv2.resize(sample, (self.reference.shape[1], self.reference.shape[0]), interpolation=cv2.INTER_AREA)

        sample_gray = to_gray(sample)
        orb = self._orb_score(sample_gray)
        ssim = _compute_ssim(self.reference_gray, sample_gray)
        hist = _hist_corr_bgr(self.reference, sample)
        phash = _phash_similarity(self.reference_gray, sample_gray)

        # Blend structure, keypoints, color and perceptual hash to improve robustness.
        if orb > 0.0:
            score = 0.40 * orb + 0.30 * ssim + 0.15 * hist + 0.15 * phash
        else:
            score = 0.45 * ssim + 0.30 * hist + 0.25 * phash

        return SimilarityResult(
            score=float(np.clip(score, 0.0, 1.0)),
            orb_score=orb,
            ssim_score=ssim,
            hist_score=hist,
            phash_score=phash,
        )
