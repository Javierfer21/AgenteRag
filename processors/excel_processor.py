"""
Excel document processor using pandas and openpyxl.
"""
import io

from processors.base import BaseProcessor


class ExcelProcessor(BaseProcessor):
    """Extracts content from Excel files (.xlsx, .xls), handling multiple sheets."""

    def process(self, file_bytes: bytes) -> str:
        """Read all sheets from an Excel file and return their string representations.

        Args:
            file_bytes: Raw bytes of the Excel file.

        Returns:
            String with content from each sheet, including basic statistics.

        Raises:
            ImportError: If pandas or openpyxl are not installed.
            Exception: If the Excel file cannot be parsed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for Excel processing. "
                "Install it with: pip install pandas openpyxl"
            ) from exc

        excel_buffer = io.BytesIO(file_bytes)

        try:
            all_sheets: dict = pd.read_excel(
                excel_buffer,
                sheet_name=None,  # Read all sheets
                engine="openpyxl",
            )
        except Exception:
            # Retry without specifying engine (handles .xls with xlrd if available)
            excel_buffer.seek(0)
            all_sheets = pd.read_excel(excel_buffer, sheet_name=None)

        output_parts: list[str] = []

        for sheet_name, df in all_sheets.items():
            lines = [
                f"=== Hoja: {sheet_name} ===",
                f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}",
                f"Columnas: {', '.join(str(c) for c in df.columns.tolist())}",
                "",
                "Vista previa (primeras 5 filas):",
                df.head().to_string(index=False),
            ]

            numeric_df = df.select_dtypes(include="number")
            if not numeric_df.empty:
                lines += [
                    "",
                    "Estadísticas numéricas:",
                    numeric_df.describe().to_string(),
                ]

            lines += [
                "",
                "Datos completos (CSV):",
                df.to_csv(index=False),
            ]

            output_parts.append("\n".join(lines))

        return "\n\n".join(output_parts)
