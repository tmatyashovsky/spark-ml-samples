package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.MnistService;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/mnist")
public class MnistController {

    @Autowired
    private MnistService mnistService;

    @RequestMapping(value = "/train", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> trainMnistClassificationModel() {
        Map<String, Object> modelStatistics = mnistService.trainImages();
        return new ResponseEntity<>(modelStatistics, HttpStatus.OK);
    }

    @RequestMapping(value = "/predict", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> predictDigit() {
        Map<String, Object> predictions = mnistService.predict();
        return new ResponseEntity<>(predictions, HttpStatus.OK);
    }

    @RequestMapping(value = "/convert", method = RequestMethod.GET)
    ResponseEntity convert() {
        mnistService.convertMnist();
        return new ResponseEntity<>(HttpStatus.OK);
    }

}
