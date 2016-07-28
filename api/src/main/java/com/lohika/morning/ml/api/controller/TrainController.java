package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.CatDogService;
import com.lohika.morning.ml.api.service.LyricsService;
import com.lohika.morning.ml.api.service.MnistService;
import com.lohika.morning.ml.api.service.DouService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TrainController {

    @Autowired
    private DouService douService;

    @Autowired
    private CatDogService catDogService;

    @Autowired
    private MnistService mnistService;

    @Autowired
    private LyricsService lyricsService;

    @RequestMapping(value = "/train_cats_dogs", method = RequestMethod.GET)
    ResponseEntity trainCatDogClassificationModel() {
        catDogService.useSVM();
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @RequestMapping(value = "/train_mnist", method = RequestMethod.GET)
    ResponseEntity trainMnistClassificationModel() {
        mnistService.useLogisticRegression();
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @RequestMapping(value = "/train_dou_regression", method = RequestMethod.GET)
    ResponseEntity trainDouLinearRegression() {
        douService.useLinearRegression();
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @RequestMapping(value = "/train_dou_clustering", method = RequestMethod.GET)
    ResponseEntity trainDouKMeans() {
        douService.useKMeans();
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @RequestMapping(value = "/train_lyrics", method = RequestMethod.GET)
    ResponseEntity trainDarkLyrics() {
        lyricsService.classifyLyrics();
        return new ResponseEntity<>(HttpStatus.OK);
    }

}
