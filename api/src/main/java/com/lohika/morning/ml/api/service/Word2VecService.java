package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.lyrics.word2vec.Similarity;
import com.lohika.morning.ml.spark.driver.service.lyrics.word2vec.Synonym;
import com.lohika.morning.ml.spark.driver.service.lyrics.pipeline.Word2VecPipeline;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class Word2VecService {

    @Autowired
    private Word2VecPipeline word2VecLyricsPipeline;

    public Map<String, Object> train() {
        return word2VecLyricsPipeline.train();
    }

    public List<Synonym> findSynonyms(String lyrics) {
        return word2VecLyricsPipeline.findSynonyms(lyrics);
    }

    public List<Similarity> calculateSimilarity(String lyrics) {
        return word2VecLyricsPipeline.calculateSimilarity(lyrics);
    }
}
