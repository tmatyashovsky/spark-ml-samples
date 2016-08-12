package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.lyrics.TextService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class LyricsService {

    @Value("${lyrics.training.set.directory.path}")
    private String lyricsTrainingSetDirectoryPath;

    @Autowired
    private TextService textService;

    public void classifyLyrics(int vectorSize, int sentencesInVerse) {
        textService.classifyDarkLyrics(lyricsTrainingSetDirectoryPath, vectorSize, sentencesInVerse);
    }

}
