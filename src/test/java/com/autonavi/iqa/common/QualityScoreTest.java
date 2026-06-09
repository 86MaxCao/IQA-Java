package com.autonavi.iqa.common;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class QualityScoreTest {

    @Test
    void successfulScoreStoresValueAndUrl() {
        QualityScore qs = new QualityScore(3.5, "http://example.com/img.jpg");

        assertEquals(3.5, qs.getScore(), 1e-9);
        assertEquals("http://example.com/img.jpg", qs.getImageUrl());
        assertTrue(qs.isSuccess());
        assertFalse(qs.isError());
        assertNull(qs.getErrorMessage());
    }

    @Test
    void errorScoreSetsErrorFields() {
        QualityScore qs = new QualityScore("http://example.com/bad.jpg", "download failed");

        assertEquals(QualityScore.ERROR_SCORE, qs.getScore(), 1e-9);
        assertEquals("http://example.com/bad.jpg", qs.getImageUrl());
        assertFalse(qs.isSuccess());
        assertTrue(qs.isError());
        assertEquals("download failed", qs.getErrorMessage());
    }

    @Test
    void errorScoreConstantIsNegativeOne() {
        assertEquals(-1.0, QualityScore.ERROR_SCORE, 1e-9);
    }

    @Test
    void scoreRangeConstants() {
        assertEquals(1.0, QualityScore.MIN_SCORE, 1e-9);
        assertEquals(5.0, QualityScore.MAX_SCORE, 1e-9);
        assertTrue(QualityScore.MIN_SCORE < QualityScore.MAX_SCORE);
    }

    @Test
    void toStringContainsScoreForSuccess() {
        QualityScore qs = new QualityScore(4.2, "http://img.png");
        String s = qs.toString();
        assertTrue(s.contains("4.2"));
        assertTrue(s.contains("http://img.png"));
        assertFalse(s.contains("error"));
    }

    @Test
    void toStringContainsMessageForError() {
        QualityScore qs = new QualityScore("http://img.png", "timeout");
        String s = qs.toString();
        assertTrue(s.contains("timeout"));
        assertTrue(s.contains("http://img.png"));
    }

    @Test
    void zeroScoreIsNotAnError() {
        QualityScore qs = new QualityScore(0.0, "url");
        assertTrue(qs.isSuccess());
        assertFalse(qs.isError());
    }

    @Test
    void negativeOneScoreViaConstructorIsError() {
        QualityScore qs = new QualityScore(-1.0, "url");
        assertTrue(qs.isSuccess()); // success constructor always sets success=true
        assertTrue(qs.isError());   // but isError() checks score == ERROR_SCORE
    }
}
