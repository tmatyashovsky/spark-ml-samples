package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.Genre;
import com.lohika.morning.ml.api.service.LyricsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class PredictionController {

    @Autowired
    private LyricsService lyricsService;

    @RequestMapping(value = "/predict_genre", method = RequestMethod.POST)
    ResponseEntity<String> trainCatDogClassificationModel(@RequestBody String unknownLyrics) {
        Genre genre = lyricsService.predictGenre(unknownLyrics);

        return new ResponseEntity<>(genre.getValue(), HttpStatus.OK);
    }

}
