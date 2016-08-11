package com.lohika.morning.ml.spark.distributed.library.function.map;

import java.util.ArrayList;
import java.util.List;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import scala.collection.mutable.WrappedArray;

public class VerseMapFunction implements MapFunction<Row, Row> {

    @Override
    public Row call(Row input) throws Exception {
        WrappedArray<String> verses = ((WrappedArray<String>)input.getAs("verses")).thisCollection();

        List<String> verse = new ArrayList<>();

        for (int i=0; i<verses.length(); i++) {
            String sentence = verses.apply(i);
            String[] words = sentence.split(" ");

            for (int j=0; j<words.length; j++) {
                verse.add(words[j]);
            }
        }

        return RowFactory.create(verse.toArray());
    }

}
