from pydantic import BaseModel, Field
from typing import Optional


class CourseExtractionResult(BaseModel):
    course_code: Optional[str] = Field(default=None, description="Course code, e.g. 410252")
    course_name: Optional[str] = Field(default=None, description="Full course name")

    lecture_hours_per_week: Optional[float] = Field(default=None)
    practical_hours_per_week: Optional[float] = Field(default=None)
    tutorial_hours_per_week: Optional[float] = Field(default=None)

    theory_credits: Optional[float] = Field(default=None)
    practical_credits: Optional[float] = Field(default=None)
    tutorial_credits: Optional[float] = Field(default=None)
    total_credits: Optional[float] = Field(default=None)

    in_semester_exam_marks: Optional[int] = Field(default=None)
    end_semester_exam_marks: Optional[int] = Field(default=None)
    term_work_marks: Optional[int] = Field(default=None)
    practical_oral_marks: Optional[int] = Field(default=None)
    total_marks: Optional[int] = Field(default=None)

    extraction_notes: Optional[str] = Field(default=None)