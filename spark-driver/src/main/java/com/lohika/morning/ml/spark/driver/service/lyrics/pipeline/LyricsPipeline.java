package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.util.Map;
import org.apache.spark.ml.tuning.CrossValidatorModel;

public interface LyricsPipeline {

    CrossValidatorModel classify();

    GenrePrediction predict(String unknownLyrics);

    Map<String, Object> getModelStatistics(CrossValidatorModel model);

}
