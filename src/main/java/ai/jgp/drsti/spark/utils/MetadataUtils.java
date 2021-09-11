package ai.jgp.drsti.spark.utils;

import org.apache.spark.sql.types.StructField;

public class MetadataUtils {

  public static String explain(StructField field) {
    StringBuilder sb = new StringBuilder();

    sb.append("Name ....... ");
    sb.append(field.name());
    sb.append("\nMetadata ... ");
    sb.append(field.metadata());
    sb.append("\nType ....... ");
    sb.append(field.dataType());

    return sb.toString();
  }

}
