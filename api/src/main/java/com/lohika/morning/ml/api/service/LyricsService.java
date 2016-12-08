package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import com.lohika.morning.ml.spark.driver.service.lyrics.TextService;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class LyricsService {

    @Value("${lyrics.training.set.directory.path}")
    private String lyricsTrainingSetDirectoryPath;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    @Autowired
    private TextService textService;

    public Map<String, Object> classifyLyrics() {
        return textService.classifyLyricsUsingLogisticRegression(
            lyricsTrainingSetDirectoryPath, lyricsModelDirectoryPath);
    }

    public GenrePrediction predictGenre(String unknownLyrics) {
        return textService.predict(unknownLyrics, lyricsModelDirectoryPath);
    }

}
