package com.lohika.morning.ml.spark.driver.service.lyrics.transformer;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Uniter extends Transformer implements MLWritable {

    private String uid;

    public Uniter(String uid) {
        this.uid = uid;
    }

    public Uniter() {
        this.uid = "Uniter" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> words) {
        // Unite words into a sentence again.
        Dataset<Row> stemmedSentences = words.groupBy(Column.ID.getName(), Column.ROW_NUMBER.getName(), Column.LABEL.getName())
                        .agg(functions.concat_ws(" ", functions.collect_list(Column.STEMMED_WORD.getName())).as(Column.STEMMED_SENTENCE.getName()));
        stemmedSentences.cache();
        stemmedSentences.count();

        return stemmedSentences;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                Column.ID.getStructType(),
                Column.ROW_NUMBER.getStructType(),
                Column.LABEL.getStructType(),
                Column.STEMMED_SENTENCE.getStructType()
        });
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Uniter> read() {
        return new DefaultParamsReader<>();
    }

}
