
-- Table creation
create table TrashCan (canID varchar(8) not null, location varchar(50), primary key (canID));
create table Trash (trashName varchar(30) not null, categoryName varchar(30), primary key (trashName));
create table Bag (canID varchar(8) not null, trashName varchar(30) not null, quantity int, primary key (canID, trashName), foreign key (canID) references TrashCan (canID), foreign key (trashName) references Trash (trashName));

-- Populate Tables
insert into TrashCan values ('1', 'kitchen');

insert into Trash values ('apple', 'compostable'), ('sandwich', 'compostable'), ('banana', 'compostable'), ('orange', 'compostable'), ('broccoli', 'compostable'), ('carrot', 'compostable'), ('pizza', 'compostable'), ('hot dog', 'compostable'), ('donut', 'compostable'), ('cake', 'compostable'), ('bottle', 'recyclable'), ('fork', 'recyclable'), ('spoon', 'recyclable'), ('knife', 'recyclable'), ('bowl', 'trash'), ('cup', 'recyclable'), ('other', 'other');

insert into Bag values ('1', 'apple', '0'), ('1', 'sandwich', '0'), ('1', 'banana', '0'), ('1', 'orange', '0'), ('1', 'broccoli', '0'), ('1', 'carrot', '0'), ('1', 'pizza', '0'), ('1', 'hot dog', '0'), ('1', 'donut', '0'), ('1', 'cake', '0'), ('1', 'bottle', '0'), ('1', 'fork', '0'), ('1', 'spoon', '0'), ('1', 'knife', '0'), ('1', 'bowl', '0'), ('1', 'cup', '0'), ('1', 'other', '0');