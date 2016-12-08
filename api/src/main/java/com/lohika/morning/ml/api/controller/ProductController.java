package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.ProductService;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/products")
public class ProductController {

    @Autowired
    private ProductService productService;

    @RequestMapping(value = "/train", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> trainForSalaryPrediction() {
        Map<String, Object> trainStatistics = productService.trainRandomForestClassifier();

        return new ResponseEntity<>(trainStatistics, HttpStatus.OK);
    }
}
