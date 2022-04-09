package com.lohika.morning.ml.spark.driver.service.lyrics;

public class GenrePrediction {

    private String genre;
    private Double metalProbability; // TODO Senthuran This is the custom genre
    private Double popProbability;
    private Double countryProbability;
    private Double bluesProbability;
    private Double jazzProbability;
    private Double reggaeProbability;
    private Double rockProbability;
    private Double hiphopProbability;

    public GenrePrediction(String genre, 
                            Double metalProbability,
                            Double popProbability,
                            Double countryProbability,
                            Double bluesProbability,
                            Double jazzProbability,
                            Double reggaeProbability,
                            Double rockProbability,
                            Double hiphopProbability) {
        this.genre = genre;
        this.metalProbability = metalProbability; // TODO Senthuran This is the custom genre
        this.popProbability = popProbability;
        this.countryProbability = countryProbability;
        this.bluesProbability = bluesProbability;
        this.jazzProbability = jazzProbability;
        this.reggaeProbability = reggaeProbability;
        this.rockProbability = rockProbability;
        this.hiphopProbability = hiphopProbability;
    }

    public GenrePrediction(String genre) {
        this.genre = genre;
    }

    public String getGenre() {
        return genre;
    }

    public Double getMetalProbability() { // TODO Senthuran This is the custom genre
        return metalProbability;
    }

    public Double getPopProbability() {
        return popProbability;
    }

    public Double getCountryProbability() {
        return countryProbability;
    }

    public Double getBluesProbability() {
        return bluesProbability;
    }

    public Double getJazzProbability() {
        return jazzProbability;
    }

    public Double getReggaeProbability() {
        return reggaeProbability;
    }

    public Double getRockProbability() {
        return rockProbability;
    }

    public Double getHiphopProbability() {
        return hiphopProbability;
    }
}
