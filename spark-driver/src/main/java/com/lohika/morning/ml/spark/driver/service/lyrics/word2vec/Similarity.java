package com.lohika.morning.ml.spark.driver.service.lyrics.word2vec;

public class Similarity {

    private String verse1;
    private String verse2;
    private Double cosine;

    public Similarity(String verse1, String verse2, Double cosine) {
        this.verse1 = verse1;
        this.verse2 = verse2;
        this.cosine = cosine;
    }

    public String getVerse1() {
        return verse1;
    }

    public String getVerse2() {
        return verse2;
    }

    public Double getCosine() {
        return cosine;
    }
}
