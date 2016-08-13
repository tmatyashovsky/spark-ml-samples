package com.lohika.morning.ml.api.service;

public enum Genre {

    METAL("Metal \\m/"),

    POP("Pop <(^.^)/"),

    UNKNOWN("Don\'t know :(");

    private final String value;

    Genre(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
