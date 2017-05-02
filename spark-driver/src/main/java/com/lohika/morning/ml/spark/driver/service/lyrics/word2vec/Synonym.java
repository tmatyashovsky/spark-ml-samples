package com.lohika.morning.ml.spark.driver.service.lyrics.word2vec;

public class Synonym {

    private String verse;
    private Double cosine;

    public Synonym(String verse, Double cosine) {
        this.verse = verse;
        this.cosine = cosine;
    }

    public String getVerse() {
        return verse;
    }

    public Double getCosine() {
        return cosine;
    }
}
