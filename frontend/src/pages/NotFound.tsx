
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

const NotFound = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-primary">404</h1>
        <h2 className="text-2xl font-medium">Page Not Found</h2>
        <p className="text-muted-foreground max-w-md">
          The page you are looking for doesn't exist or has been moved.
        </p>
        <Button 
          onClick={() => navigate("/")}
          className="mt-4"
        >
          Return to Dashboard
        </Button>
      </div>
    </div>
  );
};

export default NotFound;
