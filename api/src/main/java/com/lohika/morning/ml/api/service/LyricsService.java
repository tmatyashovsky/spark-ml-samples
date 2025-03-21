package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import com.lohika.morning.ml.spark.driver.service.lyrics.pipeline.LyricsPipeline;
import java.util.Map;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.springframework.stereotype.Component;

import javax.annotation.Resource;

@Component
public class LyricsService {

    @Resource(name = "${lyrics.pipeline}")
    private LyricsPipeline pipeline;

    public Map<String, Object> classifyLyrics() {
        System.out.println("LyricsService classifyLyrics"+ pipeline);
        CrossValidatorModel model = pipeline.classify();
        return pipeline.getModelStatistics(model);
    }

    public GenrePrediction predictGenre(final String unknownLyrics) {
        return pipeline.predict(unknownLyrics);
    }

}
