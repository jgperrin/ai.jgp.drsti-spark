package ai.jgp.drsti.spark.utils;

import static org.apache.spark.sql.functions.col;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.MetadataBuilder;

public abstract class DataframeUtils {

  public static Dataset<Row> addMetadata(Dataset<Row> df, String columnName,
      String key, String value) {
    Metadata metadata = new MetadataBuilder()
        .withMetadata(ColumnUtils.getMetadata(df, columnName))
        .putString(key, value)
        .build();
    Column col = col(columnName);
    return df.withColumn(columnName, col, metadata);
  }

  public static Dataset<Row> addMetadata(Dataset<Row> df, String key,
      String value) {
    for (String colName : df.columns()) {
      df = addMetadata(df, colName, key, value);
    }
    return df;
  }

  public static Object min(Dataset<Row> df, String columnName) {
    return df.selectExpr("MIN(" + columnName + ")").first().get(0);
  }

  public static Object max(Dataset<Row> df, String columnName) {
    return df.selectExpr("MAX(" + columnName + ")").first().get(0);
  }

  public static int maxAsInt(Dataset<Row> df, String columnName) {
    return ((Long) max(df, columnName)).intValue();
  }

}
