package com.lohika.morning.ml.spark.distributed.library.function.map.lyrics;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.tartarus.snowball.SnowballStemmer;

public class StemmingFunction implements MapFunction<Row, Row> {

    private static SnowballStemmer stemmer = initializeStemmer();

    private static SnowballStemmer initializeStemmer () {
        try {
            Class stemClass = Class.forName("org.tartarus.snowball.ext.englishStemmer");

            return (SnowballStemmer) stemClass.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    @Override
    public Row call(Row input) throws Exception {
        stemmer.setCurrent(input.getAs("filteredWord"));
        stemmer.stem();
        String stemmedWord = stemmer.getCurrent();

        return RowFactory.create(
                input.getAs("value"),
                input.getAs("label"),
                input.getAs("clean"),
                input.getAs("id"),
                input.getAs("rowNumber"),
                input.getAs("words"),
                input.getAs("filteredWords"),
                input.getAs("filteredWord"),
                stemmedWord);
    }

}
