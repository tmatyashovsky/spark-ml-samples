package com.lohika.morning.ml.api.service;

public enum Level {

    JUNIOR("Junior Engineer"),

    MIDDLE("Middle Engineer"),

    SENIOR("Senior Engineer"),

    UNKNOWN("Unknown");

    private final String value;

    Level(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
