package com.lohika.morning.ml.spark.driver.service.lyrics;

public class GenrePrediction {

    private String genre;
    private Double metalProbability;
    private Double popProbability;

    public GenrePrediction(String genre, Double metalProbability, Double popProbability) {
        this.genre = genre;
        this.metalProbability = metalProbability;
        this.popProbability = popProbability;
    }

    public GenrePrediction(String genre) {
        this.genre = genre;
    }

    public String getGenre() {
        return genre;
    }

    public Double getMetalProbability() {
        return metalProbability;
    }

    public Double getPopProbability() {
        return popProbability;
    }
}
