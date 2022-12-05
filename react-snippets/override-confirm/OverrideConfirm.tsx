`
Problem:
- Modal shows if entered value breaches the threshold.
  "Confirmed!" text only show if entered value is within threshold

OR

- User can override and confirm anyway.
  The handleConfirm onClick logic is shared between Parent and Child.
- Assume we can't just pass setIsConfirmed to child because the parent logic
  is complex and we don't want to repeat ourselves / pass all the variables associated
  to the parent to the child

- We can't call handleConfirm right after setting isOverridden state because state updates are asynchronous

Solution:
- The core of the solution is to useEffect to perform handleConfirm only after isOverriden state has been updated
`

import "./styles.css";
import { useState, useEffect, Dispatch, SetStateAction } from "react";
import { Stack, Button, Dialog, TextField } from "@mui/material";

interface ChildModalProps {
  isOpen: boolean;
  setIsOpen: Dispatch<SetStateAction<boolean>>;
  handleConfirm: () => void;
  isOverridden: boolean;
  setIsOverridden: Dispatch<SetStateAction<boolean>>;
}

const ChildModal = ({
  isOpen,
  setIsOpen,
  handleConfirm,
  isOverridden,
  setIsOverridden
}: ChildModalProps) => {
  // Assume we have a lot of logic and dependencies inside handleConfirm
  // and don't want to pass isConfirmed, setIsConfirmed to child
  // hence passing handleConfirm directly from parent
  const handleOverrideConfirm = () => {
    setIsOverridden(true);
    console.log(`isOverriden ${isOverridden}`);
    setIsOpen(false);
    // NOTICE if we put handleConfirm here, we have to click "Override confirm"
    // twice before it's overridden, this is because state updates are asynchronous
    // handleConfirm();
  };

  // Instead we want to handleConfirm only after isOverriden state has changed
  useEffect(() => {
    if (isOverridden) {
      console.log(`useEffect isOverriden ${isOverridden}`);
      handleConfirm();
    }
  }, [isOverridden, handleConfirm]);

  return (
    <Dialog open={isOpen}>
      <Button
        onClick={() => {
          setIsOpen(false);
        }}
      >
        Cancel
      </Button>
      <Button onClick={handleOverrideConfirm}>Override confirm</Button>
    </Dialog>
  );
};

const Parent = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [value, setValue] = useState<number>();
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [isOverridden, setIsOverridden] = useState(false);

  const handleValueChange = (e: any) => {
    setValue(e.target.value);
  };

  const handleConfirm = () => {
    if (!value) return;
    if (value > 5 && !isOverridden) {
      setIsConfirmed(false);
      setIsOpen(true);
      return;
    }
    setIsConfirmed(true);
  };

  const handleReset = () => {
    setIsConfirmed(false);
    setIsOverridden(false);
    setIsOpen(false);
  };

  return (
    <>
      <ChildModal
        isOpen={isOpen}
        setIsOpen={setIsOpen}
        handleConfirm={handleConfirm}
        isOverridden={isOverridden}
        setIsOverridden={setIsOverridden}
      />
      <Stack>
        Enter a number (Threshold: 5):
        <TextField onChange={handleValueChange}>{value}</TextField>
        <Button onClick={handleConfirm}>Confirm</Button>
        <Button onClick={handleReset}>Reset</Button>
        {isConfirmed ? "confirmed!" : ""}
      </Stack>
    </>
  );
};

export default function App() {
  return (
    <div className="App">
      <Parent />
    </div>
  );
}
