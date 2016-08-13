package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.DouService;
import com.lohika.morning.ml.api.service.Level;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/dou")
public class DouController {

    @Autowired
    private DouService douService;

    @RequestMapping(value = "/salaries/train", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> trainForSalaryPrediction() {
        Map<String, Object> trainStatistics = douService.trainDOURegressionModel();

        return new ResponseEntity<>(trainStatistics, HttpStatus.OK);
    }

    @RequestMapping(value = "/salaries/predict", method = RequestMethod.GET)
    ResponseEntity<Double> predictSalary(
            @RequestParam(defaultValue = "10") Double experience,
            @RequestParam(defaultValue = "продвинутый") String englishLevel,
            @RequestParam(defaultValue = "Java") String programmingLanguage) {
        Double prediction = douService.predictSalary(experience, englishLevel, programmingLanguage);

        return new ResponseEntity<>(prediction, HttpStatus.OK);
    }

    @RequestMapping(value = "/levels/train", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> clusterizeITSpecialists(
            @RequestParam(defaultValue = "3") Integer clusters) {
        Map<String, Object> trainStatistics = douService.clusterizeITSpecialists(clusters);

        return new ResponseEntity<>(trainStatistics, HttpStatus.OK);
    }

    @RequestMapping(value = "/levels/predict", method = RequestMethod.GET)
    ResponseEntity<String> predictLevel(
        @RequestParam(defaultValue = "4000") Integer salary,
        @RequestParam(defaultValue = "10") Double experience,
        @RequestParam(defaultValue = "продвинутый") String englishLevel) {
        Level level = douService.predictLevel(salary, experience, englishLevel);

        return new ResponseEntity<>(level.getValue(), HttpStatus.OK);
    }
}
