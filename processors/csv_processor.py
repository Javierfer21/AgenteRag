"""
CSV document processor using pandas.
"""
import io

from processors.base import BaseProcessor


class CSVProcessor(BaseProcessor):
    """Extracts content and basic statistics from CSV files using pandas."""

    def process(self, file_bytes: bytes) -> str:
        """Read a CSV file and return its string representation with stats.

        Args:
            file_bytes: Raw bytes of the CSV file.

        Returns:
            String containing the CSV data and descriptive statistics.

        Raises:
            ImportError: If pandas is not installed.
            Exception: If the CSV cannot be parsed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for CSV processing. "
                "Install it with: pip install pandas"
            ) from exc

        # Try UTF-8 first, then Latin-1 as fallback
        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # If both fail, return raw decoded text
            return file_bytes.decode("latin-1", errors="replace")

        lines = [
            f"CSV con {df.shape[0]} filas y {df.shape[1]} columnas.",
            f"Columnas: {', '.join(df.columns.tolist())}",
            "",
            "Vista previa (primeras 5 filas):",
            df.head().to_string(index=False),
            "",
            "Estadísticas descriptivas:",
            df.describe(include="all").to_string(),
            "",
            "Datos completos (CSV):",
            df.to_csv(index=False),
        ]
        return "\n".join(lines)
