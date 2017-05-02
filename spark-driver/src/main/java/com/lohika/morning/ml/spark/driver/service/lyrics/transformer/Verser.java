package com.lohika.morning.ml.spark.driver.service.lyrics.transformer;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Option;

public class Verser extends Transformer implements MLWritable {

    private String verseId = "verseId";
    private String uid;

    public Verser(String uid) {
        this.uid = uid;
    }

    public Verser() {
        this.uid = "Verser" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        Dataset<Row> verses = sentences.withColumn(
                verseId,
                functions.floor(functions.column(Column.ROW_NUMBER.getName()).minus(1).divide(getSentencesInVerse())).plus(1)
        );

        verses = verses.groupBy(Column.ID.getName(), verseId).agg(
                functions.first(Column.LABEL.getName()).as(Column.LABEL.getName()),
                functions.split(functions.concat_ws(" ",
                                functions.collect_list(
                                        functions.column(Column.STEMMED_SENTENCE.getName()))), " ").as(Column.VERSE.getName())
        );

        return verses.drop(Column.ID.getName()).drop(verseId);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                Column.LABEL.getStructType(),
                Column.VERSE.getStructType()
        });
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return uid;
    }

    public IntParam sentencesInVerse() {
        return new IntParam(uid, "sentencesInVerse", "");
    }

    public Integer getSentencesInVerse() {
        final Option<Object> sentencesInVerse = get(sentencesInVerse());
        return sentencesInVerse.isEmpty() ? 1 : (Integer) sentencesInVerse.get();
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Verser> read() {
        return new DefaultParamsReader<>();
    }

}
