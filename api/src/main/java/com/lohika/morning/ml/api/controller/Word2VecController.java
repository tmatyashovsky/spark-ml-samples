package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.Word2VecService;
import com.lohika.morning.ml.spark.driver.service.lyrics.word2vec.Similarity;
import com.lohika.morning.ml.spark.driver.service.lyrics.word2vec.Synonym;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/word-to-vec")
public class Word2VecController {

    @Autowired
    private Word2VecService word2VecService;

    @RequestMapping(value = "/train", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> trainWord2VecModel() {
        Map<String, Object> trainStatistics = word2VecService.train();

        return new ResponseEntity<>(trainStatistics, HttpStatus.OK);
    }

    @RequestMapping(value = "/synonyms", method = RequestMethod.POST)
    ResponseEntity<List<Synonym>> findSynonyms(@RequestBody String lyrics) {
         List<Synonym> synonyms = word2VecService.findSynonyms(lyrics);

        return new ResponseEntity<>(synonyms, HttpStatus.OK);
    }

    @RequestMapping(value = "/similarity", method = RequestMethod.POST)
    ResponseEntity<List<Similarity>> calculateSimilarity(@RequestBody String lyrics) {
        List<Similarity> similarities = word2VecService.calculateSimilarity(lyrics);

        return new ResponseEntity<>(similarities, HttpStatus.OK);
    }

}
