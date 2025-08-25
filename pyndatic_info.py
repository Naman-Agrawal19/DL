# EmailStr and AnyHttpUrl are custom data types provided by pydantic for data validation
from datetime import date
from pydantic import BaseModel, ValidationError, EmailStr, AnyUrl, Field, field_validator   
from typing import List, Dict, Optional, Annotated

class Person(BaseModel):
    name: str = Field(min_length=3, max_length=30) # name is a field with validation rules like minimum length 3 and maximum length 20
    age: int = Field(default=18, ge=18, le=90) # age is a field with default value 18 and validation rules like greater than or equal to 18 and less than or equal to 90
    gender: str='Not Defined'
    hobbies: Optional[List[str]] = Field(default=None, max_length=5) # hobbies is a field with default value None and validation rule like maximum length 5
    contact: Dict[str, str]
    email: EmailStr # EmailStr is a custom data type provided by pydantic for data validation
    profile: AnyUrl # AnyUrl is a custom data type provided by pydantic for data validation

def enter_details(detail: Person):
    print(f"Name: {detail.name}")
    print(f"Age: {detail.age}")
    print(f"Gender: {detail.gender}")
    print(f"Hobbies: {detail.hobbies}")
    print(f"Contact: {detail.contact}")
    print(f"Email: {detail.email}")
    print(f"Profile: {detail.profile}")
    print("Thank you for entering details")

try:
    person1 = Person(name="Naman", gender="Male", hobbies=["Reading", "Coding"], contact={"phone": "1234567890"}, email="naman@gmail.com", profile="https://www.linkedin.com/naman")
except ValidationError as e:
    print("Validation Error")
enter_details(person1)



try:
    person2 = Person(name="Rahul", age='thirty', gender="Male", hobbies=["Reading", "Coding"], contact={"phone": "1234567890"}, email="rahul@gmail.com", profile="https://www.linkedin.com/rahul")
except ValidationError as e:
    print("Validation Error")
 # this will throw an error because age is not an integer


try:
    person3 = Person(name="Kavi",age=18, contact={"phone": "1234567890"}, email="kavi@gmail.com", profile="http://linkedin.co.in/kavi")
except ValidationError as e:
    print(e)
enter_details(person3)


try:
    person4 = Person(name="Ravi",age=10, contact={"phone": "1234567890"}, hobbies=['a', 'b', 'c', 'd', 'e', 'f'], email="kavi@gmail.com", profile="http://linkedin.co.in/kavi")
except ValidationError as e:
    print(e)
# this will throw an error because hobbies has more than 5 elements and age is less than 18




## --------------------------------------------
## now adding metadata using Annotations

class student(BaseModel):
    name: Annotated[str, Field(min_length=3, max_length=30, title="Name of the student", description="Enter the name of the student", examples=["Naman", "Rahul", "Kavi"])]
    weight: Annotated[float, Field(default=20, ge=20, title="Weight of the student", description="Enter the weight of the student", examples=[20, 18, 15])]

def enter_student(detail: student):
    print(f"Name: {detail.name}")
    print(f"Weight: {detail.weight}")
    print("Thank you for entering details")

student1 = student(name="Krishna", weight='20')
enter_student(student1)
# this will run because pydantic is smart enough to understand that weight is a float and use type coercion to convert it to float

# if we want to suppress the type coercion we can use the raw parameter
class teacher(BaseModel):
    name: Annotated[str, Field(min_length=3, max_length=30, title="Name of the student", description="Enter the name of the student", examples=["Naman", "Rahul", "Kavi"])]
    age: Annotated[int, Field(default=20, ge=20, title="Age of the teacher", strict=True,description="Enter the age of the teacher", examples=[20, 25, 30])]

def enter_teacher(detail: teacher):
    print(f"Name: {detail.name}")
    print(f"Age: {detail.age}")
    print("Thank you for entering details")
teacher1 = teacher(name="Krishna", age='20')
# this will throw an error because age is not an integer and type coercion is suppressed using strict=True


## --------------------------------------------
## now let say we want to check if the email contains gmail or not, we can do that using field_validator
from pydantic  import field_validator
class employee(BaseModel):
    name: Annotated[str, Field(min_length=3, max_length=30, title="Name of the employee", description="Enter the name of the employee", examples=["Naman", "Rahul", "Kavi"])]
    email: Annotated[EmailStr, Field(title="Email of the employee", description="Enter the email of the employee", examples=["naman@gmail.com", "rahul@gmail.com", "kavi@yahoo.com"])]
    age: Annotated[int, Field(default=20, ge=20, le=100,title="Age of the employee", strict=True,description="Enter the age of the employee", examples=[20, 25, 30])]

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        value = value.lower()
        if "@gmail.com" not in value:
            raise ValueError("Email must contain gmail ")
        return value
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        return " ".join(word.capitalize() for word in value.split())


def enter_employee(detail: employee):
    print(f"Name: {detail.name}")
    print(f"Email: {detail.email}")
    print("Thank you for entering details")

employee1 = employee(name="naman agr", email="NAMAN@gmail.com")
enter_employee(employee1)

employee2 = employee(name="naman agr", email="NAMAN@yahoo.com")
# this will throw an error because email must contain gmail 

## now, there is a parameter called mode in pydantic, which can be used to specify the mode of validatoin like to validate before or after the type coercion

class animal(BaseModel):
    age: Annotated[int, Field(default=20, title="Age of the animal", strict=True,description="Enter the age of the animal", examples=[20, 25, 30])]
    weight: Annotated[float, Field(default=20, ge=10, title="Weight of the animal", description="Enter the weight of the animal", examples=[20, 18, 15])]

    @field_validator("age", mode="before")
    @classmethod
    def validate_age(cls, value):
        if value < 0:
            raise ValueError("Age must be greater than 0")
        else:
            print("Age is valid")
        return value
    
    @field_validator("weight", mode="after") # default mode is after
    @classmethod
    def validate_weight(cls, value):
        if value < 0:
            raise ValueError("Weight must be greater than 0")    
        else:
            print("Weight is valid")
        return value
    
animal1 = animal(age='10', weight=20) 
# this will throw an error because age is string type and mode is before, so condition is checked before type coercion

animal2 = animal(age=10, weight='20')
# this will work because mode is after and condition is checked after type coercionq


## --------------------------------------------
## using model_validator to validate multiple fields
from pydantic import model_validator, computed_field

class Employee_again(BaseModel):
    name: Annotated[str, Field(min_length=3, max_length=30, title="Name of the employee", description="Enter the name of the employee", examples=["Naman", "Rahul", "Kavi"])]

    is_termianted: Annotated[bool, Field(title="Is the employee termintated", description="Enter the is the employee termintated", examples=[True, False])]

    email: Annotated[EmailStr, Field(default="not_defined", title="Email of the employee", description="Enter the email of the employee", examples=["naman@gmail.com", "rahul@gmail.com", "kavi@yahoo.com"])]

    dob: Annotated[date, Field(ge=date(1970, 1, 1), le=date.today(), title="Date of birth of the employee", description="Enter the date of birth of the employee", examples=["2022-01-01", "2022-02-02", "2022-03-03"])]

    @model_validator(mode="after")
    @classmethod
    def validate_employee(cls, value):
        value.email = value.email.lower()
        if value.is_termianted==False and value.email == "not_defined":
            raise ValueError("Email must be defined if employee is not termintated")
        else:
            print("Done")
        print(value)
        return value
    
    @computed_field
    @property
    def calc_age(self) -> int:
        todays = date.today()
        age = todays.year - self.dob.year - ((todays.month, todays.day) < (self.dob.month, self.dob.day))
        return age

    

employee_again1 = Employee_again(name="naman agr", is_termianted=True, dob="2022-01-01")
employee_again2 = Employee_again(name="naman agr", is_termianted=False, dob="2023-01-10") # this will throw an error because email must be defined if employee is not termintated
employee_again3 = Employee_again(name="naman agr", is_termianted=False, email="NAMAN@gmail.com", dob="1997-08-07") 
employee_again4 = Employee_again(name="naman agr", is_termianted=False, email="NAMAN@gmail.com", dob="2029-01-01")
# this will throw an error because dob is not in the range of 1970-01-01 to current date

## -------------------------------------------------
## Nested Model

class Address(BaseModel):
    address1: str
    city: str
    state: str
    pincode: int = Field(title="Pincode of the address", description="Enter the pincode of the address", examples=[110059, 110060, 110061])

    @field_validator("pincode")
    @classmethod
    def validate_pincode(cls, value):
        if len(str(value)) != 6:
            raise ValueError("Wrong pincode format")
        return value

class People(BaseModel):
    name: str
    age: int = 18
    address: Address

def enter_people(detail: People):
    print(f"Name: {detail.name}")
    print(f"Age: {detail.age}")
    print(f"Address: {detail.address}")
    print(f"Pincode: {detail.address.pincode}")
    print("Thank you for entering details")

address_dict1 = Address(address1="123", city="Delhi", state="NCR", pincode=11005) # this will throw an error because pincode is not in the correct format

address_dict2 = Address(address1="123", city="Delhi", state="NCR", pincode=110059)

people2 = People(name="Rahul", age=25, address=address_dict2)
enter_people(people2)

people3 = People(name="Naman", address=address_dict2)
enter_people(people3)

# dumping the model
temp = people2.model_dump()
print(temp)
print(type(temp))

print(people2.model_dump_json(), '\n', type(people2.model_dump_json()))

# let say i want to exclude the pincode from the model dump
temp = people2.model_dump(exclude={'address': {'pincode'}})
print(temp)

# also we can prevent dumping the unsetted values
temp = people3.model_dump(exclude_unset=True)
print(temp) # as gender was default value in this, it will not be dumped

