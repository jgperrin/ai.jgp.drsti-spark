package ai.jgp.drsti.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import ai.jgp.drsti.spark.utils.DataframeUtils;

public abstract class DrstiUtils {

  public static Dataset<Row> setHeader(
      Dataset<Row> df,
      String columnName,
      String header) {
    return DataframeUtils.addMetadata(
        df, columnName, DrstiK.HEADER, header);
  }

}
